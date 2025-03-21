﻿
__kernel void computeSADCosts(__global const uchar* leftImage,   // Left image (grayscale)
    __global const uchar* rightImage,  // Right image (grayscale)
    __global float* costFunction,     // Output cost function
    const int width,              // Width of the images
    const int height,             // Height of the images
	const int halfWindowSize,         // Half the window size
    const int disparityRange) {       // Disparity range


    int idx = get_global_id(0);
    int x = idx % width;
    int y = idx / width;

    // Ensure we stay within the image boundaries
    if (x >= width || y >= height) {
        return;
    }
    
    // Iterate over the disparity range for the current pixel
    for (int d = 0; d < disparityRange; ++d) {
        // Compute the x-coordinate for the right image based on disparity
        int rightX = x - d;

        // Check boundary conditions for right image
        if (rightX >= 0) {
            // Compute the SAD for the current disparity
            float sad = 0.0f;

            // Calculate the SAD by comparing corresponding pixels in the left and right image
            for (int dy = -halfWindowSize; dy <= halfWindowSize; ++dy) {
                for (int dx = -halfWindowSize; dx <= halfWindowSize; ++dx) {
                    int leftPixel = leftImage[(y + dy) * width + (x + dx)];
                    int rightPixel = rightImage[(y + dy) * width + (rightX + dx)];
                    sad += fabs((float)(leftPixel - rightPixel));
                }
            }

            // Store the SAD in the cost function array for this pixel and disparity
            costFunction[(y * width + x) * disparityRange + d] = sad;
        }
        else {
            // Assign a high cost value if the right image is out of bounds
            costFunction[(y * width + x) * disparityRange + d] = FLT_MAX;
        }
    }
}

#define TILE_SIZE 32  
#define MAX_DISPARITY 64

__kernel void horizontalAggregation(
    __global const float* costFunction,
    __global float* aggregatedCost,
    const int width,
    const int height,
    const int disparityRange,
    const float P1,
    const float P2,
    const int tileIndex
    ) {

    int y = get_global_id(1);
	int threadId = get_local_id(0); // Local x-coordinate within the tile
	int group_x = tileIndex * TILE_SIZE; // Starting x-coordinate of the tile (group_x + threadId = x)

    if (y >= height) return;

    __local float costTile[TILE_SIZE][MAX_DISPARITY];
    __local float aggTile[TILE_SIZE + 1][MAX_DISPARITY];

	__local float minCostSynchronizationBuffer[TILE_SIZE];

    int disparityStart = (get_local_id(0) * disparityRange) / get_local_size(0);
    int disparityEnd = ((get_local_id(0) + 1) * disparityRange) / get_local_size(0);

    float minCostPrevX = FLT_MAX;
    float minCostCurrX = FLT_MAX;
    for (int d = disparityStart; d < disparityEnd; d++) {
		// load the prefix column of the tile from the previous tile
        aggTile[0][d] = aggregatedCost[(y * width + (group_x - 1)) * disparityRange + d];
		minCostCurrX = fmin(minCostCurrX, aggTile[0][d]);
		for (int x = group_x; x < group_x + TILE_SIZE; x++) {
			int xInTile = x - group_x;
			costTile[xInTile][d] = costFunction[(y * width + x) * disparityRange + d];
			aggTile[xInTile + 1][d] = 0;
		}
    }
    minCostSynchronizationBuffer[threadId] = minCostCurrX;
    // min reduction of minCostCurrX
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < TILE_SIZE; i++) {
        minCostCurrX = fmin(minCostCurrX, minCostSynchronizationBuffer[i]);
    }
    

    barrier(CLK_LOCAL_MEM_FENCE);


	// Iterate over the tiles in the row
    for (int x = group_x; x < group_x + TILE_SIZE && x < width; x++) {
		// group_x: The starting x-coordinate of the tile
		// xInTile: The x-coordinate of the current pixel within the tile (between 0 and TILE_SIZE)
        int xInTile = x - group_x; 
        if (xInTile >= TILE_SIZE) break;  // Prevent out-of-bounds

        int offsetX = (y * width + x) * disparityRange;
        // perform aggregation but store results in the local memory
        minCostPrevX = minCostCurrX;
        minCostCurrX = FLT_MAX;

        for (int d = disparityStart; d < disparityEnd; d++) {
            float currCost = costTile[xInTile][d];
            float minCost = currCost;

            float leftCost = aggTile[xInTile + 1 - 1][d];
            minCost = leftCost;

            if (d > 0) {
                minCost = fmin(minCost, aggTile[xInTile + 1 - 1][d - 1] + P1);
            }
            if (d < disparityRange - 1) {
                minCost = fmin(minCost, aggTile[xInTile + 1 - 1][d + 1] + P1);
            }
            minCost = fmin(minCost, P2);
            minCost = minCost + currCost - minCostPrevX;
            

            aggTile[xInTile + 1][d] = minCost;
            minCostCurrX = fmin(minCostCurrX, minCost);
        }
        minCostSynchronizationBuffer[threadId] = minCostCurrX;
        // min reduction of minCostCurrX
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < TILE_SIZE; i++) {
            minCostCurrX = fmin(minCostCurrX, minCostSynchronizationBuffer[i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }



    // After processing the tile, store results to global memory
    // Only the thread responsible for the current pixel writes to global memory
	int xx = group_x + threadId;
    if (threadId < TILE_SIZE && xx < width) {
        for (int d = 0; d < disparityRange; d++) {
			int offsetX = (y * width + xx) * disparityRange;
            aggregatedCost[offsetX + d] = aggTile[threadId + 1][d];
        }
    }
}

#define INVALID_DISP 0
#define DISP_SCALE 16

__kernel void computeBestDisparity(__global float* aggregatedCost,   // Input aggregated cost function
    __global ushort* disparityMap,  // Output best disparity map
    const int width,            // Width of the image
    const int height,           // Height of the image
    const int disparityRange,   // Disparity range
    float uniquenessRatio    ) {    

    int idx = get_global_id(0);
    int x = idx % width;
    int y = idx / width;


    // Ensure we stay within the image boundaries
    if (x >= width || y >= height) {
        return;
    }

    // Initialize the minimum cost to a large value
    float minCost = FLT_MAX;
    int bestDisparity = -1;

	int costIndexOffset = (y * width + x) * disparityRange;
    // Iterate over all disparity values for the current pixel
    for (int d = 0; d < disparityRange; ++d) {
        // Get the aggregated cost for the current disparity (x, y, d)
        float cost = aggregatedCost[costIndexOffset + d];

        // If the cost is lower than the current minimum, update the best disparity
        if (cost < minCost) {
            minCost = cost;
            bestDisparity = d;
        }
    }
    // discard pixels with too much uncertainty (pixels for which the second 
    // non-neighboring best disparity is too close to the best one)
    for (int d = 0; d < disparityRange; d++) {

        float cost = aggregatedCost[costIndexOffset + d];
        if ((minCost > cost * uniquenessRatio) && (abs(bestDisparity - d) > 1)) {
            bestDisparity = INVALID_DISP;
        }
    }


	if (bestDisparity > 0 && bestDisparity < disparityRange - 1) {
		float c0 = aggregatedCost[costIndexOffset + bestDisparity - 1];
		float c1 = aggregatedCost[costIndexOffset + bestDisparity];
		float c2 = aggregatedCost[costIndexOffset + bestDisparity + 1];
		float w0 = 1.0f / (fabs(c1 - c0) + 1.0f);
		float w2 = 1.0f / (fabs(c1 - c2) + 1.0f);
		float subpixelOffset = (w2 - w0) / (w0 + w2);
		bestDisparity = bestDisparity * DISP_SCALE + subpixelOffset * DISP_SCALE;
    }
    else {
		bestDisparity = INVALID_DISP;
    }


    // Store the best disparity (as unsigned short) for the current pixel
    disparityMap[y * width + x] = (ushort)bestDisparity;
}



__kernel void computeSAD(
    __global const uchar* leftImage,  // Left image (grayscale)
    __global const uchar* rightImage, // Right image (grayscale)
    __global unsigned short* disparityMap,  // Output disparity map
    const int width,                  // Image width
    const int height,                 // Image height
    const int maxDisparity,           // Maximum disparity to test
    const int windowSize)             // Window size for the SAD
{
    int idx = get_global_id(0); 
	int x = idx % width;
	int y = idx / width;
    int minSAD = INT_MAX;
    int bestDisparity = 0;

    // Loop over all possible disparities
    for (int d = 0; d < maxDisparity; d++) {
        int sad = 0;

        // Calculate SAD in a window around (x, y)
        for (int dx = -windowSize; dx <= windowSize; dx++) {
            for (int dy = -windowSize; dy <= windowSize; dy++) {
                int leftX = x + dx;
                int leftY = y + dy;
                int rightX = leftX - d;

                if (leftX >= 0 && leftX < width && leftY >= 0 && leftY < height && rightX >= 0 && rightX < width) {
                    // Calculate absolute difference between corresponding pixels
                    int leftPixel = leftImage[leftY * width + leftX];
                    int rightPixel = rightImage[leftY * width + rightX];
                    sad += abs(leftPixel - rightPixel);
                }
            }
        }

        // Update the best disparity based on minimum SAD
        if (sad < minSAD) {
            minSAD = sad;
            bestDisparity = d;
        }
    }
        
    // Store the best disparity in the output map
	disparityMap[y * width + x] = bestDisparity * 16;
    
}



