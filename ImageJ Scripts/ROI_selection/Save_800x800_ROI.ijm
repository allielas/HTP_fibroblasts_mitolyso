macro "ROI assignment [7] "   //Name this as you wish. Assigns '7' as shortcut to run macro.
{
//Getting set up for the session
	if(nImages == 0)
	{
		waitForUser("Open an image if there is not one open");
	}
	dir = File.directory;
	filelist = getFileList(dir);  //This directs the program to the folder to use.
	run("Select None"); //Removes any ROIs that may have been left from the previous image.
	original=getTitle();	//returns string with the image title, which is the original image. 
	
				//ID'g it here so it can be called again at the end in order to move to the next image in the folder
//Preparing area to be measured
	for (i = 1; i <= nSlices; i++) {
    	setSlice(i);
    	run("Enhance Contrast", "saturated=0.35");
    }
	makeRectangle(686, 683, 800, 800);
	waitForUser("Move your ROI");
	roiManager("add & draw");
	run("Crop");

	saveAs("Tiff", dir + "/crops/" + original +"_crop.tif");
//Go to next image in folder and start again:
	//selectWindow(original);	//This selects the original image, with the filename so that 'Open next' will have a reference of what was previously open so it knows where to go next.
	run("Open Next");	//This becomes the active image for the next session
	close("\\Others");	//This closes all the inactive image(s) without closing the Results window or the ROI Manager
	waitForUser("Hit 7 to begin working on the next image");
}