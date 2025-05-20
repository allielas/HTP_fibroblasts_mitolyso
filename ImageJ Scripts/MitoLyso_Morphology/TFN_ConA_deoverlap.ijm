

function iterateFolder(input,output) {
	//batchmode helps a lot
	setBatchMode(true);
	
	list = getFileList(input);
	channelDistance = list.length / 4; // calculate the distance btwn channels
	
	for (i = 0; i < list.length; i++){
		if (i >= channelDistance) {
	        removeBleedthrough(input, output, list[i], list[i-channelDistance]);
		}
	}
	setBatchMode(false);
}

input = "/mnt/bigdisk1/Allie S/Replicative_Age_Project/MRC-5_Max_Projection_Images/20240313/";
output = "/mnt/bigdisk1/Allie S/Replicative_Age_Project/IJ Projection script/De_overlapped_Tfn/";
// specify I/O
iterateFolder(input,output)


function removeBleedthrough(input, output, filename, prevFilename) {
        open(input + filename);
        
        // Modify the condition to include only the row/cols that have ConA-594 and Tfn-A647
        if(filename.contains("ch3") && (filename.contains("c07") || filename.contains("c08") || filename.contains("c09") || filename.contains("c10")|| filename.contains("c11"))) {
        	
        	open(input + prevFilename);
	        // select the file and multiply by the constant
	        selectImage(filename);
	        run("Multiply...", "value=8.2974787");
			// subtract the ConA image from the intensity-adjusted Tfn image
			imageCalculator("Subtract create", filename, prevFilename);
			selectImage("Result of " + filename);
	        saveAs("tif", output + "De-Overlapped" + filename);
        }
        close();
}


/*
open("/mnt/bigdisk1/Allie S/Replicative_Age_Project/MRC-5_Max_Projection_Images/20240301/MAX_ch3-r02c07f13.tif");

selectImage("MAX_ch3+*");
open


imageCalculator("Subtract create", "MAX_ch3-r02c07f13.tif","MAX_ch2-r02c07f13.tif");
*/
