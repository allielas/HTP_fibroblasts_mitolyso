function processChannel(folder,out_ch_max,out_ch_sum,base,channel) {
	run("Image Sequence...", "open=["+folder+"] number=7 starting=1 increment=1 scale=100 or="+base+"p0[0-9]-ch"+channel+".* sort");
	// make the sequence match the number of slices
	run("Z Project...", "start=1 stop=7 projection=[Max Intensity]");
	saveAs("Tiff", out_ch_max);
	close();
	
	selectWindow("Images");
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
        out_ch3_sum = result_folder + "SUM_ch3-" + base + ",tif";
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

in_folder = "/media/mattiazzilab/Mattiazzi#9/AllieS/20241112_AS_MRC-5_Rep4__2024-11-12T16_55_52-Measurement 1/Images";
out_folder ="/mnt/bigdisk1/AllieSpangaro/Morphology_Replicative_Age_Project/SUM_test"+"/"; //output path
filename = "r02c02f01p01-ch1sk1fk1fl1.tiff"
processTiff(in_folder,filename,out_folder);