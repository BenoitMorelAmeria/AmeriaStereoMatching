
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

#define TOTAL_TILE_SIZE 48  
#define EFFECTIVE_TILE_SIZE 32
#define LEFT_TILE_OFFSET (TOTAL_TILE_SIZE - EFFECTIVE_TILE_SIZE)
#define MAX_DISPARITY 64

// each working group computes the aggregation for a tile
// the tile consists of:
// - the effective tile (one pixel per work item)
// - the left offset tile (a few pixels before the effective tile)

__kernel void horizontalAggregation(
    __global const float* costFunction,
    __global float* aggregatedCost,
    const int width,
    const int height,
    const int disparityRange,
    const float P1,
    const float P2) {

    int y = get_global_id(1);
	int threadId = get_local_id(0);
	int tileId = get_group_id(0);
	bool useLeftTile = tileId > 0; // the first tile does not have a left tile

    /*
	int local_x = get_local_id(0); // Local x-coordinate within the tile
	int workGroupIndex = get_group_id(0); 
	int group_x = tileIndex * TILE_SIZE; // Starting x-coordinate of the tile (group_x + local_x = x)
    */
    if (y >= height) return;

    __local float costTile[TOTAL_TILE_SIZE][MAX_DISPARITY];
    __local float aggTile[TOTAL_TILE_SIZE][MAX_DISPARITY];
	__local float minCostSynchronizationBuffer[TOTAL_TILE_SIZE];

    
	// Initialize the local memory
    // we have to copy the cost function for the whole tile
    for (int d = 0; d < disparityRange; d++) {
		// left tile initialization
		if (useLeftTile && threadId < LEFT_TILE_OFFSET) {
			int x = (tileId - 1) * EFFECTIVE_TILE_SIZE + threadId;
			costTile[threadId][d] = costFunction[(y * width + x) * disparityRange + d];
			// initialize the first column of the aggregation tile
			aggTile[threadId][d] = (threadId == 0) ? costTile[0][d] : 0; 
		}
        
		// effective tile initialization
		int x = tileId * EFFECTIVE_TILE_SIZE + threadId;
        if (x < width) {
            costTile[threadId][d] = costFunction[(y * width + x) * disparityRange + d];
			// Initialize the first column of the aggregation tile if there is no left tile
            aggTile[threadId][d] = (!useLeftTile && threadId == 0) ? costTile[threadId][d] : 0;
        }
    }
    

    barrier(CLK_LOCAL_MEM_FENCE);
    
    float minCostPrevX = FLT_MAX;
    float minCostCurrX = FLT_MAX;

	// Iterate over the pixels in the whole tile
	// Each item in the work group executes the same code
    int startX = tileId * EFFECTIVE_TILE_SIZE;
    int endX = startX + EFFECTIVE_TILE_SIZE;
	if (useLeftTile) {
		startX -= LEFT_TILE_OFFSET;
	}

    
    for (int x = startX; x < endX && x < width; x++) {
		int xInTile = x - startX;
        int offsetX = (y * width + x) * disparityRange;


        int disparityStart = (get_local_id(0) * disparityRange) / get_local_size(0);
        int disparityEnd = ((get_local_id(0) + 1) * disparityRange) / get_local_size(0);


        // perform aggregation but store results in the local memory
        minCostPrevX = minCostCurrX;
        minCostCurrX = FLT_MAX;
        
        for (int d = disparityStart; d < disparityEnd; d++) {
            float currCost = costTile[xInTile][d];
            float minCost = currCost;

            if (xInTile > 0) {
                float leftCost = aggTile[xInTile - 1][d];
                minCost = leftCost;

                if (d > 0) {
                    minCost = fmin(minCost, aggTile[xInTile - 1][d - 1] + P1);
                }
                if (d < disparityRange - 1) {
                    minCost = fmin(minCost, aggTile[xInTile - 1][d + 1] + P1);
                }
                minCost = fmin(minCost, P2);
                minCost = minCost + currCost - minCostPrevX;
            }

            aggTile[xInTile][d] = minCost;
            minCostCurrX = fmin(minCostCurrX, minCost);
        }
        
        minCostSynchronizationBuffer[threadId] = minCostCurrX;
        // min reduction of minCostCurrX
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < EFFECTIVE_TILE_SIZE; i++) {
            minCostCurrX = fmin(minCostCurrX, minCostSynchronizationBuffer[i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    

    
    // After processing the tile, store the effective tile (only!) to global memory
    // Only the thread responsible for the current pixel writes to global memory
    int xx = tileId * EFFECTIVE_TILE_SIZE + threadId;
    
    if (threadId < EFFECTIVE_TILE_SIZE && xx < width) {
        for (int d = 0; d < disparityRange; d++) {
			int offsetX = (y * width + xx) * disparityRange;
            aggregatedCost[offsetX + d] = aggTile[threadId][d];
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



