
// Used for Allie's Thesis 2023-2024

function manualAccess(path, key) {

	files = getFileList(path);
	cur_path = path + files[key];
	print(cur_path);
	open(cur_path);
}

//manualAccess('/media/mattiazzilab/Mattiazzi #7/MattiazziLab_data/AS_MRC-5/20240206_AS_MRC5_R1t1_R2t0__2024-02-06T18_19_41-Measurement 1/Images/', 1121)


function processChannel(folder,out_ch,base,channel) {
	run("Image Sequence...", "open=["+folder+"] number=7 starting=1 increment=1 scale=100 or="+base+"p0[0-9]-ch"+channel+".* sort");
	// make the sequence match the number of slices
	run("Z Project...", "start=1 stop=7 projection=[Max Intensity]");
	saveAs("Tiff", out_ch);
	close();
	selectWindow("Images");
	close();
}

function processTiff(folder,filename,result_folder){
    print("Filename: " + filename);
    bits = split(filename, "-"); 
    base = substring(bits[0],0,9);
    channel = substring(bits[1],2,3);
    print(bits[0],"," , base, ",", channel);
    if (channel=="1") {
    	in_ch1 = folder + filename;
    	//print(in_ch1);
    	out_ch1 = result_folder + "MAX_ch1-" + base + ".tif";
    	//print (out_ch1);
    	if (File.exists(out_ch1)) {
        	return;
    	}
    	processChannel(folder,out_ch1,base,channel);
    }
    else if (channel=="2"){
    	in_ch2 = folder + filename;
    	out_ch2 = result_folder + "MAX_ch2-" + base + ".tif";
    	if (File.exists(out_ch2)) {
        	return;
    	}
    	processChannel(folder,out_ch2,base,channel);
    }
    else if (channel=="3"){
        in_ch3 = folder + filename;
        out_ch3 = result_folder + "MAX_ch3-" + base + ".tif";
        if (File.exists(out_ch3)) {
            return;
        }
        processChannel(folder,out_ch3,base,channel);
    }
    else if (channel=="4"){
        in_ch4 = folder + filename;
        out_ch4 = result_folder + "MAX_ch4-" + base + ".tif";
        if (File.exists(out_ch4)) {
            return;
        }
        processChannel(folder,out_ch4,base,channel);
    }
}


function recfolder(folder, result_folder) {
    print("recursing into " + folder);
    setBatchMode(true);
    files = getFileList(folder);
    for (i = 0; i < files.length; i++) {
        cur_path = folder + files[i];
        if(File.isDirectory(folder + files[i])) {
        	if (files[i]=="Assaylayout/" || matches(files[i], "^r08")){
        		print("Skip");
        	}
        	else {
		        res = result_folder + files[i];
		        if (!File.exists(res)) {
		            File.makeDirectory(res);
		        }
		        recfolder(cur_path, res);
		    }
        }
        else if (endsWith(files[i], ".tiff")) {
            processTiff(folder, files[i], result_folder);
        }
    }
    setBatchMode(false);
}
//#@ File(style="directory") in_folder;
//#@ File(style="directory") out_folder;

in_folder = "/media/mattiazzilab/Mattiazzi#9/AllieS/20250205_AS_MRC-5_AnitbodyTest_v2__2025-02-05T19_47_33-Measurement 1/Images"; //input path
out_folder ="/mnt/bigdisk1/AllieSpangaro/Senesence_Classification/Projection_Images/AntibodyTest_R1"+"/"; //output path
recfolder(in_folder, out_folder);
print("Done");
run("Quit");

//eval("script", "System.exit(0);");

//Run full script with: </path/to/Fiji.app/ImageJ-linux64> --headless -macro </path/to/macro.ijm> 'input=</path/to/input/folder> output=</path/to/output/folder>'

//On workstation /home/mattiazzilab/Downloads/Fiji.app/ImageJ-linux64 --headless -macro /mnt/bigdisk2/Allie_Spangaro/MRC-5_Max_Projections/Script/fileopener.ijm 'input=/media/mattiazzilab/Mattiazzi #7/MattiazziLab_data/AS_MRC-5/20240206_AS_MRC5_R1t1_R2t0__2024-02-06T18_19_41-Measurement 1/Images/ output=/mnt/bigdisk2/Allie_Spangaro/MRC-5_Max_Projections'
