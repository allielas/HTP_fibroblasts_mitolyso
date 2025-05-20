// ImageJ macro for segmenting a list of images
directory = "/Volumes/AllieS/Microscopy Images/20241119_MRC5_Senescence_Optimization_R2/40x/DMSO/";
//VSI version: change for other ofrmats

filelist = getFileList(directory) 
for (i = 0; i < lengthOf(filelist); i++) {
    if (endsWith(filelist[i], ".vsi")) { 
        open(directory + File.separator + filelist[i]);
        //Make a composite of a 3 channel image with beta gal bg (GFP), DAPI, and DIC
        run("Make Composite");
        //Stack.setActiveChannels("011");
		run("Next Slice [>]");
		run("Enhance Contrast", "saturated=0.35");
		run("Cyan");
		run("Next Slice [>]");
		run("Enhance Contrast", "saturated=0.35");
		run("Grays");
        waitForUser("Do your thing before opening the next image");
    } 
}
