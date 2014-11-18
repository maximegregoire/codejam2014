using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FaceFinder
{
    class Program
    {
        private const string RESOURCE_PATH = "resources";
        private const string FACE_CLASSIFIER_PATH = RESOURCE_PATH + @"\faceClassifier.xml";
        private const string DEFAULT_INPUT_FOLDER = "input";
        private const string DEFAULT_OUTPUT_FOLDER = "output";

        static void Main(string[] args)
        {
            string inputFolder = DEFAULT_INPUT_FOLDER;
            string outputFolder = DEFAULT_OUTPUT_FOLDER;

            if (args.Length >= 1) inputFolder = args[0];
            if (args.Length >= 2) outputFolder = args[1];

            FindFaces(inputFolder, outputFolder);
        }

        static void FindFaces(string inputFolder, string outputFolder)
        {
            Directory.CreateDirectory(outputFolder);

            foreach (var file in Directory.GetFiles(inputFolder, "*.gif"))
            {
                var colorImage = new Image<Rgb, byte>(file);
                colorImage.ROI = FindFaceRect(colorImage);

                var grayFace = new Image<Gray, byte>(colorImage.Bitmap);

                grayFace.Save(Path.Combine(outputFolder, Path.GetFileName(file)));
            }
        }

        static Rectangle FindFaceRect(Image<Rgb, byte> image)
        {
            //Find the first red byte indicative of the face border
            int i = 0;
            var b = image.Bytes;
            for (; i < b.Length && (b[i] != 255 || b[i+1] != 0 || b[i+2] != 0); i += 3) ;

            if (i == b.Length) throw new ArgumentException("Image does not contain a face rectangle!");

            //Register top left corner of face rect;
            var faceRect = new Rectangle();
            faceRect.X = (i/3) % image.Width;
            faceRect.Y = (i/3) / image.Width;

            //Find width of face rect
            for (; b[i] == 255 && b[i + 1] == 0 && b[i + 2] == 0; i += 3) ++faceRect.Width;
            
            //Find height of face rect
            i -= 3;
            for (; b[i] == 255 && b[i + 1] == 0 && b[i + 2] == 0; i += image.Width * 3) ++faceRect.Height;

            if (faceRect.Height < 3 || faceRect.Width < 3) throw new ArgumentException("The face rectangle must contain at least one pixel!");

            //Return only the interior region of the rectangle
            faceRect.Height -= 2;
            faceRect.Width -= 2;
            faceRect.X += 1;
            faceRect.Y += 1;

            return faceRect;
        }
    }
}
