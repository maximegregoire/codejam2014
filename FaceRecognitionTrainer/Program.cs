using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Database;

namespace FaceRecognitionTrainer
{
    class Program
    {
        private const string DEFAULT_PLAIN_INPUT_FOLDER = "inputPlain";
        private const string DEFAULT_MARKED_INPUT_FOLDER = "inputMarked";
        private const string DEFAULT_OUTPUT_FOLDER = "output";

        static void Main(string[] args)
        {
            string plainInputFolder = DEFAULT_PLAIN_INPUT_FOLDER;
            string markedInputFolder = DEFAULT_MARKED_INPUT_FOLDER;
            string outputFolder = DEFAULT_OUTPUT_FOLDER;

            if (args.Length >= 1) plainInputFolder = args[0];
            if (args.Length >= 2) markedInputFolder = args[1];
            if (args.Length >= 3) outputFolder = args[2];

            var database = ProcessImages(plainInputFolder, markedInputFolder, outputFolder);
            Serializer.WriteDatabase(outputFolder, database);

            foreach (var file in Directory.GetFiles("photos", "*.bmp"))
            {
                var identification = Match(file, database);
                Console.WriteLine(Path.GetFileName(file) + " " + identification);
            }
        }

        static Dictionary<int, SubjectModelDataCollection<byte>> ProcessImages(string plainInputFolder, string markedInputFolder, string outputFolder)
        {
            Directory.CreateDirectory(outputFolder);

            var database = new Dictionary<int, SubjectModelDataCollection<byte>>();

            foreach (var fileNameWithoutExt in Directory.GetFiles(plainInputFolder, "*.bmp").Select(f => Path.GetFileNameWithoutExtension(f)))
            {
                string plainImagePath = Path.Combine(plainInputFolder, fileNameWithoutExt) + ".bmp";
                string markedImagePath = Path.Combine(markedInputFolder, fileNameWithoutExt) + ".bmp";
                string outputKeypointImage = Path.Combine(outputFolder, fileNameWithoutExt) + ".bmp";

                if (!File.Exists(markedImagePath))
                {
                    throw new ArgumentException("The plain picture " + plainImagePath + " does not have a corresponding marked picture.");
                }

                //Find the regions we are interested in, marked by red rectangles, in the marked picture
                var markedImage = new Image<Rgb, byte>(markedImagePath);
                var faceRects = FindFaceRects(markedImage);

                //if (faceRects.Count != 2) throw new ArgumentException("Invalid amount of face rectangles. Should be 2.");

                //Creat a mask corresponding to these regions
                var mask = CreateMask(faceRects, markedImage.Height, markedImage.Width);

                //Extract keypoints for these regions only
                var plainImage = new Image<Gray, byte>(plainImagePath);

                //var fast = new FastDetector(1, true);
                //var orb = new ORBDetector(500);
                var brisk = new Brisk(1, 3, 1.0f);
                //var surf = new SURFDetector(500, true);
                var keypoints = brisk.DetectKeyPointsRaw(plainImage, mask);

                //For debugging purposes, draw the keypoints to an actual image and save it
                var o = Features2DToolbox.DrawKeypoints(plainImage, keypoints, new Bgr(Color.Blue), Features2DToolbox.KeypointDrawType.DRAW_RICH_KEYPOINTS);
                o.Save(outputKeypointImage);

                //Get descriptors
                //var freak = new Freak(true, true, 22.0f, 3);
                var brief = new BriefDescriptorExtractor();
                var descriptors = brisk.ComputeDescriptorsRaw(plainImage, null, keypoints);

                //Add the gathered data to the database
                int subjectId = Convert.ToInt32(fileNameWithoutExt.Split('_')[0]);

                if (!database.ContainsKey(subjectId)) database.Add(subjectId, new SubjectModelDataCollection<byte>()
                    {
                        subjectId = subjectId, models = new List<ModelData<byte>>()
                    }
                );

                database[subjectId].models.Add(new ModelData<byte> { keypoints = keypoints.ToArray(), descriptors = descriptors });
            }

            return database;
        }

        static List<Rectangle> FindFaceRects(Image<Rgb, byte> image)
        {
            //Find the first red byte indicative of the face border
            int i = 0;
            var b = image.Bytes;

            var faceRects = new List<Rectangle>();

            while (i < b.Length)
            {
                for (; i < b.Length && (b[i] != 255 || b[i + 1] != 0 || b[i + 2] != 0); i += 3) ;

                if (i == b.Length) continue;

                //Register top left corner of face rect;
                var faceRect = new Rectangle();
                faceRect.X = (i / 3) % image.Width;
                faceRect.Y = (i / 3) / image.Width;

                //Find width of face rect
                for (; b[i] == 255 && b[i + 1] == 0 && b[i + 2] == 0; i += 3) ++faceRect.Width;

                //Find height of face rect
                i -= 3;
                for (; b[i] == 255 && b[i + 1] == 0 && b[i + 2] == 0; i += image.Width * 3) ++faceRect.Height;

                if (faceRect.Height < 3 || faceRect.Width < 3) throw new ArgumentException("The face rectangle must contain at least one pixel!");

                i -= (faceRect.X + faceRect.Width - 1) * 3;

                //Return only the interior region of the rectangle
                faceRect.Height -= 2;
                faceRect.Width -= 2;
                faceRect.X += 1;
                faceRect.Y += 1;

                faceRects.Add(faceRect);
            }

            return faceRects;
        }

        static Image<Gray, byte> CreateMask(List<Rectangle> rects, int height, int width)
        {
            var mask = new Image<Gray, byte>(width, height, new Gray(0));

            foreach (var r in rects)
            {
                mask.ROI = r;
                mask.SetValue(255);
            }

            mask.ROI = new Rectangle(0, 0, width, height);
            return mask;
        }

        static int Match(string fileName, Dictionary<int, SubjectModelDataCollection<byte>> database)
        {
            var observedImage = new Image<Gray, byte>(fileName);

            //var fast = new FastDetector(1, true);
            //var orb = new ORBDetector(500);
            //var surf = new SURFDetector(500, true);
            var brisk = new Brisk(1, 3, 1.0f);
            //var freak = new Freak(true, true, 22.0f, 1);
            //var brief = new BriefDescriptorExtractor();

            var kpObserved = brisk.DetectKeyPointsRaw(observedImage, null);
            var descObserved = brisk.ComputeDescriptorsRaw(observedImage, null, kpObserved);

            Features2DToolbox.DrawKeypoints(observedImage, kpObserved, new Bgr(Color.Blue), Features2DToolbox.KeypointDrawType.DRAW_RICH_KEYPOINTS).Save("img.bmp");

            foreach (var subjectId in database.Keys)
            {
                var models = database[subjectId].models;

                float sum = 0;
                float nb = 0;
                float max = 0;

                foreach (var model in models)
                {
                    Matrix<int> indices;
                    Matrix<byte> mask;

                    //Try to match the features
                    BruteForceMatcher<byte> matcher = new BruteForceMatcher<byte>(DistanceType.Hamming);
                    matcher.Add(model.descriptors);

                    indices = new Matrix<int>(descObserved.Rows, 10);
                    using (Matrix<float> dist = new Matrix<float>(descObserved.Rows, 10))
                    {
                        matcher.KnnMatch(descObserved, indices, dist, 10, null);
                        mask = new Matrix<byte>(dist.Rows, 1);
                        mask.SetValue(255);
                        Features2DToolbox.VoteForUniqueness(dist, 0.7, mask);
                    }
                    
                    var kpModel = new VectorOfKeyPoint();
                    kpModel.Push(model.keypoints);
                    int numOfKeypoints = CvInvoke.cvCountNonZero(mask);
                    if (numOfKeypoints > 0)
                    {
                        numOfKeypoints = Features2DToolbox.VoteForSizeAndOrientation(kpModel, kpObserved, indices, mask, 1.5, 10);
                    }

                    sum += numOfKeypoints;
                    max = Math.Max(max, numOfKeypoints);
                    nb++;
                }

                float average = sum / nb;
            }
            

            return 0;
        }
    }
}
