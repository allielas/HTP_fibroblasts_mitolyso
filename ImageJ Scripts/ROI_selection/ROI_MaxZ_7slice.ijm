macro "ROI crop with maxZ [7] "{   //Assigns '7' as shortcut to run macro.
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
	run("Z Project...", "projection=[Max Intensity]"); //run("Z Project...", "projection=[Max Intensity]");
	zproj_name = getTitle();
	for (i = 1; i <= nSlices; i++) {
    	setSlice(i);
    	run("Enhance Contrast", "saturated=0.35");
    	//run("Enhance Local Contrast (CLAHE)", "blocksize=127 histogram=256 maximum=3 mask=*None* fast_(less_accurate) process_as_composite");
    	if(i==2){
    		setMinAndMax(100, 8000); // Set min and max to a set value for channel 2 (GFP) (used 200, 1000 for p16)
    	}
    }
	makeRectangle(686, 683, 800, 800);
	waitForUser("Move your ROI");
	roiManager("add & draw");
	run("Crop");
	//save(dir + "/crops/" + zproj_name +"_crop.tif");
	save("/Volumes/AllieS/Microscopy Images/crops_hela/" + zproj_name +"_crop.tif");
//Go to next image in folder and start again:
	//selectWindow(original);	//This selects the original image, with the filename so that 'Open next' will have a reference of what was previously open so it knows where to go next.
	selectWindow(original);
	run("Open Next");	//This becomes the active image for the next session
	close("\\Others");	//This closes all the inactive image(s) without closing the Results window or the ROI Manager
	waitForUser("Hit 7 to begin working on the next image");
}