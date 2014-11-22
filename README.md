Our submission for the 2014 edition of the CodeJam.

This is a program that identifies and match a persons face using a database.

It is made in C# and uses OpenCV (EMGU, the .NET wrapper)

To run the program, simply add the EMGU DLLs (too big for GitHub) 
in the executable folder (either Debug or Release) and use it in the
following fashion:

Add the images of subjects in the database folder (which should be created in
either Debug or Release folder), titled numberOfsubjet_pictureNumber.bmp, gif or jpeg.

Build the solution.

Run facerecognition.exe with the path of the picture to identify:
ex: facerecognition C:/photos/unknownPerson.gif
