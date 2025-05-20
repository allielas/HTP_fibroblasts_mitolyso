
#@ File (style="directory") imageFolder;
dir = File.directory;
//print(dir)
filelist = getFileList(dir); 
for (i = 0; i < lengthOf(filelist)/3; i++) {
    if (endsWith(filelist[i], ".tif")) { 
        filename = filelist[i];
        rowcolfield = substring(filename, lastIndexOf(filename, "-")+1, lastIndexOf(filename, "."));
        //print(rowcolfield);
        merge_ch(dir,rowcolfield);
        for (j = 1; j <= nSlices; j++) {
    		setSlice(j);
    		run("Enhance Contrast", "saturated=0.35");
   		}
		makeRectangle(686, 683, 800, 800);
		waitForUser("Select Croppping Region");
		run("Crop");
		saveAs("Tiff", dir + "/crops/" + filelist[i] +"_crop.tif");
		close("*");
    } 
}

function merge_ch(path,rowcolfield) {
	close("*");
	open(path + "MAX_ch1-" + rowcolfield + ".tif");
	ch_1 = getTitle();
	
	open(path + "MAX_ch2-" + rowcolfield + ".tif");
	ch_2 = getTitle();
	
	open(path + "MAX_ch3-" + rowcolfield + ".tif");
	ch_3 = getTitle();
	
	run("Merge Channels...", "c1="+ ch_2 +" c2="+ ch_1 +" c3="+ ch_3 +" create");
	rename(rowcolfield);
}
