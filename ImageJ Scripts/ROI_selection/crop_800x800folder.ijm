#@ File (style="directory") imageFolder;
dir = File.directory;
filelist = getFileList(dir); 
for (i = 0; i < lengthOf(filelist); i++) {
    if (endsWith(filelist[i], ".vsi")) { 
        open(dir + File.separator + filelist[i]);
        run("Enhance Contrast", "saturated=0.35");
		run("Next Slice [>]");
		run("Enhance Contrast", "saturated=0.35");
		run("Next Slice [>]");
		run("Enhance Contrast", "saturated=0.35");
		run("Next Slice [>]");
		run("Enhance Contrast", "saturated=0.35");
		makeRectangle(686, 683, 800, 800);
		waitForUser("Select Croppping Region");
		run("Crop");
		saveAs("Tiff", dir + "/crops/" + filelist[i] +"_crop.tif");
		close("*");
    } 
}
