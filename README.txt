14/04/2019




--- Directories ---

<<< Training >>>

Training contains the code for training the GoogleNet neural network
2 files must be provided : 
	train.txt and test.txt
Each containing lines under the format:
	{Anchor} # {negative} # {positive}
Being 3 images, with anchor and positive representing the path to 2 images of the same object and negative the path to a different object





<<< Tracking >>>

Contains the code for perfmoring the parcel tracking

--- File formats ---

Detection is not incorporated inside the programme. Please provide the following:
	- A directory containing the frames on which to perform the tracking
	- The frames must be named under the format xxxx.jpg with xxxx going from 0000 to 9999 (ex: 0018) and be in JPG format
	- Provide in the directory a txt filed called identif.txt containing the detection information
	- Detection information must be on the following format:

		{framei}.jpg:{PARCEL1} {PARCEL2}.... {PARCELn}
		{framei+1}.jpg:{PARCEL1} {PARCEL2}.... {PARCELn}
		...
		With {PARCELi} as: {colis.{parcel_id} [{top_lef_y}, {top_left_x}, {bottom_right_y}, {bottom_right_x}]; }

		Example:
 			"0031.jpg:colis.0001 [200, 0, 369, 141]; colis.0002 [606, 0, 752, 195]; "

	- In case there are no specific labels (ie not for evaluating the mot metric), provide a random parcel_id (may be the same for all)


The weights of the network are storred inside the data/colis_google.pth file



--- Running the code ---

In the file "sequences.txt"
Provide the name of the root directory of the sequences
Provide the name of the sequences you wish to launch (ie name of directory containing the frame).


All sequences directory must be in the same root directory.
Run the following shell command inside the root directory of the project:
	
	python "eval_mot.py" $param2 $param2 $param3

$param 1 : 1 to show images, 0 otherwise
$param 2 : 1 to save images, 0 otherwise
$param 3 : 1 to compute metrics, 0 otherwise