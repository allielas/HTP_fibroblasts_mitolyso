/*
 * Macro to open a specified image from a folder using a rXXcYYfZZ 96-well plate coordinate scheme 
 */
 

macro "OpenMaxProjImage_Phenix"{
#@ File (label = "Input directory", style = "directory") input
#@ String (label = "File suffix", value = ".tif") suffix
#@ int (value=1, min=1, max=8, style="slider") row 
#@ int (value=1, min=1, max=12, style="slider") col 
#@ int (value=1, min=1, max=40, style="slider") field
#@ String(label="What is your fluorescence channel order", description="Enter colors in order seperated by commas e.g. green,red,far red,blue") channelOrder
#@ boolean (label = "Open random image instead?") randomFlag

	//add the extra slash to satisfy linux path
	input = input+ "/";
	
	channelColours = split(channelOrder,",");
	nChannels = lengthOf(channelColours);
	//print(nChannels);
	
	if(randomFlag == 1){
		//open random if flagged, else 
		openRandomImage(input,suffix);
	}
	else{
		openImage(input,suffix);
	}
	
	function makeCoordinateString(row,col,field) { 
	// convert the integers into the Harmony RowColField syntax
		rowString = convertNumberTo2DigitString(row);
		colString = convertNumberTo2DigitString(col);
		fieldString = convertNumberTo2DigitString(field);
		rowcolfield = "r" + rowString + "c" + colString + "f" + fieldString;
		return rowcolfield;
	}
	
	function convertNumberTo2DigitString(number) { 
	// convert a number to a two digit string, with a prefix 0 added to numbers below 10
		if (number < 10){ 
			number = "0" + number;
			}	
		return "" + number;
	}
	
	function openImage(input,suffix) { 
	// opens an image at specified rowcolfield 
		rowcolfield = makeCoordinateString(row,col,field);
		merge_ch(input,rowcolfield,suffix,nChannels);
	}
	
	
	function openRandomImage(input,suffix) { 
	// calls the random_image_rowcol function and opens the image
		rowcolfield = random_image_rowcolfield(input);
		merge_ch(input,rowcolfield,suffix,nChannels);
	}
	
	function scanFolder(input) {
		list = getFileList(input);
		list = Array.sort(list);
		for (i = 0; i < list.length; i++) {
			if(File.isDirectory(input + File.separator + list[i]))
				scanFolder(input + File.separator + list[i]);
			//if(endsWith(list[i], suffix))
				//processFile(input, output, list[i]);
		}
	}
	
	function random_image_rowcolfield(input) {
	//add a randomzie function to pick random files from the file list in the specified folder
		list = getFileList(input);
		rand = ""+round(list.length * random);
		filename = list[rand];
		rowcolfield = substring(filename, 8, 17);
		return rowcolfield;
	}
	function colour_rename(channelColours){
		//rename colours to add flexibility

		for (i = 0; i < channelColours.length; i++) {
			colour = toLowerCase(channelColours[i]);
			if (colour == "g" || colour=="gfp" || colour=="488"){
					colour = "green";
			}
			else if (colour == "r" || colour=="rfp" || colour=="cy3"){
				colour = "red";
	
			}
			else if (colour == "b" || colour == "dapi"){
				colour = "blue";
	
			}
			else if (colour == "far-red" || colour == "farred" || colour=="far red" || colour=="cy5" || colour=="647") {
				colour = "fr";
			}
			channelColours[i] = colour;
			//Array.print(channelColours);
		}
		
	}
	
	function get_colour_index(colour){
		//define colours based on your image channels, returns the index of your colour value
		colour_index=-1;
		for(i=0; i< channelColours.length; i++){
			//print(colour + " " + i+  " , "+ colour_index);
			// assign if you get the match
			if(channelColours[i]== colour){
				//print(colour + " , "+ colour_index);
				colour_index = i;
			}
		}
		
		if(colour_index==-1)
		{
			//Array.print(channelColours);
			exit("colour " + colour + " not found");
		}
		return colour_index
	}
	function merge_ch(path,rowcolfield,suffix,channels) {
		//function to open each single-channel image from that location and merge into a color composite
		close("*");
		base_filename = "MAX_ch" + "1-" + rowcolfield + suffix; //construct base file name based on phenix format
		colour_rename(channelColours);
		if(File.exists(path + base_filename)){
			for (i = 1; i <= channels; i++) {
	    		//open in order by channels
				curr_img_path = path + "MAX_ch" + i+"-" + rowcolfield + suffix;
	    		open(curr_img_path);
			}
			chNames = getList("image.titles"); //get the 0-indexed list
			
			//define colour channels mappings as ch -1 
			if(nChannels ==3) {
				run("Merge Channels...", "red="+ chNames[get_colour_index("red")] +" green="+ chNames[get_colour_index("green")] +" c5="+ chNames[get_colour_index("blue")] +" create");
			}
			else {
				run("Merge Channels...", "red=" + chNames[get_colour_index("red")] +" green="+ chNames[get_colour_index("green")] +" c5="+ chNames[get_colour_index("blue")] + " c6=" + chNames[get_colour_index("fr")] +" create");
			}
			
			//c3=blue, c4=gray, c5 = cyan, c6 = magenta
			
			//change image name to avoid confusion
			rename(rowcolfield);
		} else {
			exit("Error: File not found at the plate coordinates: " + rowcolfield + "\nin directory: \n" + path);
		}
	}
}
