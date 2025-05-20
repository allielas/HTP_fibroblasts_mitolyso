
path = "/mnt/bigdisk1/Allie_S/Replicative_Age_Project/MRC-5_Max_Projection_Images/20240313/";
rowcolfield = random_image();

merge_ch(path,rowcolfield);

//add a randomzie function to pick random nums from an array
function randomize(array) {
	n = 0;
	while(n < array[1]) {
		rand = ""+round(array.length*random);
		n = parseInt(rand);
	}
	return n;
}

function random_image() {
	row = randomize(newArray(2,3,4,5,6));
	col = randomize(newArray(2,3,4,5,6));
	field = randomize(newArray(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,40));
	
	if (field < 10) {field = "0" + field;}
	row = "0" + row;
	col = "0" + col;
	rowcolfield = "r" + row + "c" + col + "f" + field;
	return rowcolfield;
}

function merge_ch(path,rowcolfield) {
	close("*");
	open(path + "MAX_ch" + "1-" + rowcolfield + ".tif");
	ch_1 = getTitle();
	
	open(path + "MAX_ch" + "2-" + rowcolfield + ".tif");
	ch_2 = getTitle();
	
	open(path + "MAX_ch" + "3-" + rowcolfield + ".tif");
	ch_3 = getTitle();
	
	open(path + "MAX_ch" + "4-" + rowcolfield + ".tif");
	ch_4 = getTitle();
	
	run("Merge Channels...", "c1="+ ch_2 +" c2="+ ch_1 +" c3="+ ch_4 +" c6="+ ch_3 +" create");
	rename(rowcolfield);
}