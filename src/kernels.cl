
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


__kernel void horizontalAggregation(__global const float* costFunction,  // Input cost function (disparity costs for each pixel and disparity)
    __global float* aggregatedCost,  // Output aggregated cost function
    const int width,             // Width of the image
    const int height,            // Height of the image
    const int disparityRange,    // Disparity range
    const float P1,              // Penalty for small disparity changes (between neighbors)
    const float P2) {            // Penalty for large disparity changes (between neighbors)

    // Get the current global index (parallelism across pixels)
    int idx = get_global_id(0);
    int y = idx;  // Row y

    // Ensure we stay within the image boundaries
    if (y >= height) {
        return;
    }
    float minCostPrevX = FLT_MAX;
	float minCostCurrX = FLT_MAX;
    // Process pixels from xStart to xStart + xChunkSize
    for (int x = 0; x < width; ++x) {
        // Iterate over disparity values for the current pixel
		int offsetX = (y * width + x) * disparityRange;
		int offsetXPrev = (y * width + (x - 1)) * disparityRange;
        minCostPrevX = minCostCurrX;
		minCostCurrX = FLT_MAX;


		// Iterate over remaining disparities
        for (int d = 0; d < disparityRange; ++d) {
            // Initialize the minimum cost to the current cost for this pixel and disparity
            float currCost = costFunction[offsetX + d];
			float minCost = currCost;
            // If we are not at the leftmost edge, we can use the left pixel's aggregated cost
            if (x > 0) {
                // Get the previously computed aggregated cost for the left neighbor with disparity d
                float leftCost = aggregatedCost[offsetXPrev + d];

                // Update minCost considering the left neighbor's aggregated cost
                minCost = leftCost ;

                // Look at the previous disparity (d-1) and apply penalty P1
                if (d > 0) {
                    float leftCostPrevDisparity = aggregatedCost[offsetXPrev + (d - 1)];
                    minCost = fmin(minCost, leftCostPrevDisparity + P1);
                }

                // Look at the next disparity (d+1) and apply penalty P1
                if (d < disparityRange - 1) {
                    float leftCostNextDisparity = aggregatedCost[offsetXPrev + (d + 1)];
                    minCost = fmin(minCost, leftCostNextDisparity + P1);
                }
				minCost = fmin(minCost, P2);
				minCost = minCost + currCost - minCostPrevX;
            }

            // Store the computed aggregated cost for the current pixel and disparity
            aggregatedCost[offsetX + d] = minCost;
			minCostCurrX = fmin(minCostCurrX, minCost);
        }
    }
}


/*

#define TILE_SIZE 32  // Adjust based on hardware and occupancy

__kernel void horizontalAggregation(__global const float* costFunction,  // Input cost function (disparity costs for each pixel and disparity)
    __global float* aggregatedCost,  // Output aggregated cost function
    const int width,             // Width of the image
    const int height,            // Height of the image
    const int disparityRange,    // Disparity range
    const float P1,              // Penalty for small disparity changes (between neighbors)
    const float P2) {            // Penalty for large disparity changes (between neighbors)

    // Get the current global index (parallelism across pixels)
    int x = get_local_id(0) + get_group_id(0) * get_local_size(0);  // Local X index in the global image
    int y = get_group_id(1);  // Row index

    // Ensure we stay within the image boundaries (avoid reading out-of-bounds)
    if (y >= height) {
        return;
    }

    // Handle the case where width isn't a multiple of 32
    if (x >= width) {
        return;  // If out of bounds in X, exit the kernel
    }

    float minCostPrevX = FLT_MAX;
    float minCostCurrX = FLT_MAX;

    // Process pixels from xStart to xStart + xChunkSize
    for (int col = x; col < width; col += get_local_size(0)) {
        // Iterate over disparity values for the current pixel
        int offsetX = (y * width + col) * disparityRange;
        int offsetXPrev = (y * width + (col - 1)) * disparityRange;

        minCostPrevX = minCostCurrX;
        minCostCurrX = FLT_MAX;

        // Iterate over disparities
        for (int d = 0; d < disparityRange; ++d) {
            // Initialize the minimum cost to the current cost for this pixel and disparity
            float currCost = costFunction[offsetX + d];
            float minCost = currCost;

            // If we are not at the leftmost edge, use the left pixel's aggregated cost
            if (col > 0) {
                // Get the previously computed aggregated cost for the left neighbor with disparity d
                float leftCost = aggregatedCost[offsetXPrev + d];

                // Update minCost considering the left neighbor's aggregated cost
                minCost = leftCost;

                // Look at the previous disparity (d-1) and apply penalty P1
                if (d > 0) {
                    float leftCostPrevDisparity = aggregatedCost[offsetXPrev + (d - 1)];
                    minCost = fmin(minCost, leftCostPrevDisparity + P1);
                }

                // Look at the next disparity (d+1) and apply penalty P1
                if (d < disparityRange - 1) {
                    float leftCostNextDisparity = aggregatedCost[offsetXPrev + (d + 1)];
                    minCost = fmin(minCost, leftCostNextDisparity + P1);
                }

                minCost = fmin(minCost, P2);
                minCost = minCost + currCost - minCostPrevX;
            }

            // Store the computed aggregated cost for the current pixel and disparity
            aggregatedCost[offsetX + d] = minCost;
            minCostCurrX = fmin(minCostCurrX, minCost);
        }
    }
}
*/


__kernel void computeBestDisparity(__global float* aggregatedCost,   // Input aggregated cost function
    __global ushort* disparityMap,  // Output best disparity map
    const int width,            // Width of the image
    const int height,           // Height of the image
    const int disparityRange) {     // Disparity range

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

    // Iterate over all disparity values for the current pixel
    for (int d = 0; d < disparityRange; ++d) {
        // Get the aggregated cost for the current disparity (x, y, d)
        float cost = aggregatedCost[(y * width + x) * disparityRange + d];

        // If the cost is lower than the current minimum, update the best disparity
        if (cost < minCost) {
            minCost = cost;
            bestDisparity = d;
        }
    }

    // Store the best disparity (as unsigned short) for the current pixel
    disparityMap[y * width + x] = (ushort)bestDisparity * 16;
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



