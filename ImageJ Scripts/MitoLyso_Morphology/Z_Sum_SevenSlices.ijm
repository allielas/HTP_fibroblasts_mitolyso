
// Used for Allie's Thesis 2023-2024

function manualAccess(path, key) {

	files = getFileList(path);
	cur_path = path + files[key];
	print(cur_path);
	open(cur_path);
}

//manualAccess('/media/mattiazzilab/Mattiazzi #7/MattiazziLab_data/AS_MRC-5/20240206_AS_MRC5_R1t1_R2t0__2024-02-06T18_19_41-Measurement 1/Images/', 1121)


function processChannel(folder,out_ch_max,out_ch_sum,base,channel) {
	run("Image Sequence...", "open=["+folder+"] number=7 starting=1 increment=1 scale=100 or="+base+"p0[0-9]-ch"+channel+".* sort");
	// make the sequence match the number of slices
	
	run("Z Project...", "start=1 stop=7 projection=[Sum Slices]");
	saveAs("Tiff", out_ch_sum);
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
    	out_ch1_max = result_folder + "MAX_ch1-" + base + ".tif";
    	out_ch1_sum = result_folder + "SUM_ch1-" + base + ".tif"; // incorporate the sum into the functions
    	//print (out_ch1);
    	if (File.exists(out_ch1_sum)) {
        	return;
    	}
    	processChannel(folder,out_ch1_max,out_ch1_sum,base,channel);
    }
    else if (channel=="2"){
    	in_ch2 = folder + filename;
    	out_ch2_max = result_folder + "MAX_ch2-" + base + ".tif";
    	out_ch2_sum = result_folder + "SUM_ch2-" + base + ".tif";
    	if (File.exists(out_ch2_sum)) {
        	return;
    	}
    	processChannel(folder,out_ch2_max,out_ch2_sum,base,channel);
    }
    else if (channel=="3"){
        in_ch3 = folder + filename;
        out_ch3_max = result_folder + "MAX_ch3-" + base + ".tif";
        out_ch3_sum = result_folder + "SUM_ch3-" + base + ".tif";
        if (File.exists(out_ch3_sum)) {
            return;
        }
        processChannel(folder,out_ch3_max,out_ch3_sum,base,channel);
    }
    else if (channel=="4"){
        in_ch4 = folder + filename;
        out_ch4_max = result_folder + "MAX_ch4-" + base + ".tif";
        out_ch4_sum = result_folder + "SUM_ch4-" + base + ".tif";
        if (File.exists(out_ch4_sum)) {
            return;
        }
        processChannel(folder,out_ch4_max,out_ch4_sum,base,channel);
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

in_folder = "/media/mattiazzilab/Mattiazzi #8/MattiazziLab_data/Allie/20240326_AS_MRC5_R1t5_R2t4_R3t3_R4t2_R5t1_R6t0__2024-03-26T18_08_46-Measurement1/Images"; //input path
out_folder ="/mnt/bigdisk1/AllieSpangaro/Morphology_Replicative_Age_Project/MRC-5_MAX_SUM_PROJ/20240324"+"/"; //output path
recfolder(in_folder, out_folder);
print("Done");
run("Quit");
eval("script", "System.exit(0);");

//Run full script with: </path/to/Fiji.app/ImageJ-linux64> --headless -macro </path/to/macro.ijm> 'input=</path/to/input/folder> output=</path/to/output/folder>'

//On workstation 1 /home/mattiazzilab/Downloads/Fiji.app/ImageJ-linux64 --headless -macro /mnt/bigdisk2/Allie_Spangaro/MRC-5_Max_Projections/Script/fileopener.ijm 'input=/media/mattiazzilab/Mattiazzi #7/MattiazziLab_data/AS_MRC-5/20240206_AS_MRC5_R1t1_R2t0__2024-02-06T18_19_41-Measurement 1/Images/ output=/mnt/bigdisk2/Allie_Spangaro/MRC-5_Max_Projections'
