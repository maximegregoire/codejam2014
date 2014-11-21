using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Database;
using System.Diagnostics;

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

            /*
            var database = ProcessImages(plainInputFolder, markedInputFolder, outputFolder);
            Serializer.WriteDatabase(outputFolder, database);

            foreach (var file in Directory.GetFiles("photos", "*.bmp"))
            {
                var identification = Match(file, database);
                Console.WriteLine(Path.GetFileName(file) + " " + identification);
            }*/

            /*
            var surf = new FastDetector(1, true);
            var brief = new BriefDescriptorExtractor();
            var ct = new SubjectCrossTester<byte>(plainInputFolder, 2, surf, brief, DistanceType.Hamming);
            Serializer.WriteModelData(outputFolder, ct.DoCrossTestForSubject<byte>(1));
            Serializer.WriteModelData(outputFolder, ct.DoCrossTestForSubject<byte>(2));
             * */

            DoEigenTest("eigen_training_set", "photos");
            //DoEigen("eigen_training_set", @"photos\1_10.bmp");
        }

        static Dictionary<int, SubjectModelDataCollection<float>> ProcessImages(string plainInputFolder, string markedInputFolder, string outputFolder)
        {
            Directory.CreateDirectory(outputFolder);

            var database = new Dictionary<int, SubjectModelDataCollection<float>>();

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
                //var brisk = new Brisk(1, 3, 1.0f);
                var surf = new SURFDetector(100, true);
                var keypoints = surf.DetectKeyPointsRaw(plainImage, mask);

                //For debugging purposes, draw the keypoints to an actual image and save it
                var o = Features2DToolbox.DrawKeypoints(plainImage, keypoints, new Bgr(Color.Blue), Features2DToolbox.KeypointDrawType.DRAW_RICH_KEYPOINTS);
                o.Save(outputKeypointImage);

                //Get descriptors
                var freak = new Freak(true, true, 22.0f, 3);
                var brief = new BriefDescriptorExtractor();
                var descriptors = surf.ComputeDescriptorsRaw(plainImage, null, keypoints);

                //Add the gathered data to the database
                int subjectId = Convert.ToInt32(fileNameWithoutExt.Split('_')[0]);
                int modelId = Convert.ToInt32(fileNameWithoutExt.Split('_')[1]);

                if (!database.ContainsKey(subjectId)) database.Add(subjectId, new SubjectModelDataCollection<float>()
                    {
                        subjectId = subjectId,
                        models = new List<ModelData<float>>()
                    }
                );

                database[subjectId].models.Add(new ModelData<float> { keypoints = keypoints.ToArray(), descriptors = descriptors, modelId = modelId });
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

        static int Match(string fileName, Dictionary<int, SubjectModelDataCollection<float>> database)
        {
            var observedImage = new Image<Gray, byte>(fileName);

            var fast = new FastDetector(5, true);
            //var orb = new ORBDetector(500);
            var surf = new SURFDetector(100, true);
            //var brisk = new Brisk(1, 3, 1.0f);
            //var freak = new Freak(true, true, 22.0f, 1);
            //var brief = new BriefDescriptorExtractor();

            var observedKeypoints = surf.DetectKeyPointsRaw(observedImage, null);
            var observedDescriptors = surf.ComputeDescriptorsRaw(observedImage, null, observedKeypoints);

            Features2DToolbox.DrawKeypoints(observedImage, observedKeypoints, new Bgr(Color.Blue), Features2DToolbox.KeypointDrawType.DRAW_RICH_KEYPOINTS).Save("img.bmp");

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
                    BruteForceMatcher<float> matcher = new BruteForceMatcher<float>(DistanceType.L2);
                    matcher.Add(model.descriptors);

                    indices = new Matrix<int>(observedDescriptors.Rows, 10);
                    using (Matrix<float> dist = new Matrix<float>(observedDescriptors.Rows, 10))
                    {
                        matcher.KnnMatch(observedDescriptors, indices, dist, 10, null);
                        mask = new Matrix<byte>(dist.Rows, 1);
                        mask.SetValue(255);
                        Features2DToolbox.VoteForUniqueness(dist, 0.8, mask);

                        var d = new List<float>();
                        //Vote for distance low enough
                        for (int i=0; i<mask.Rows; ++i)
                        {
                            d.Add(dist[i, 0]);
                            if (dist[i, 0] > 0.35)
                            {
                                mask[i, 0] = 0;
                            }
                        }

                        var d2 = d.OrderBy(dis => dis).ToArray();
                    }
                    
                    var kpModel = new VectorOfKeyPoint();
                    kpModel.Push(model.keypoints);
                    int numOfKeypoints = CvInvoke.cvCountNonZero(mask);
                    if (numOfKeypoints > 0)
                    {
                        numOfKeypoints = Features2DToolbox.VoteForSizeAndOrientation(kpModel, observedKeypoints, indices, mask, 1.5, 10);
                    }

                    sum += numOfKeypoints;
                    max = Math.Max(max, numOfKeypoints);
                    nb++;
                }

                float average = sum / nb;
            }
            

            return 0;
        }

        static void DoEigenTest(string trainingPath, string photoPath)
        {
            var imgs = new List<Image<Gray, byte>>();
            var labels = new List<string>();

            foreach (var file in Directory.GetFiles(trainingPath, "*.bmp"))
            {
                var label = Path.GetFileName(file).Split('_')[0];
                var img = new Image<Gray, byte>(file);
                var channel0 = img[0];
                channel0._EqualizeHist();
                img[0] = channel0;
                imgs.Add(img);
                labels.Add(label);
            }

            var termCrit = new MCvTermCriteria(imgs.Count, 0.001);

            var eigen = new EigenObjectRecognizer(imgs.ToArray(), labels.ToArray(), 4500, ref termCrit);

            foreach (var file in Directory.GetFiles(photoPath, "*.bmp"))
            {
                FindFaceMultiScale(trainingPath, file, eigen);
            }
       }

        static void FindFaceMultiScale(string trainingPath, string observedImagePath, EigenObjectRecognizer eigen)
        {
            double currentScale = 0.4f;
            //double[] scales = {1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5};

            //Find the face in test picture
            var observedImage = new Image<Gray, byte>(observedImagePath);
            

            while (currentScale <= 1)
            {
                var scaledObservedImage = observedImage.Resize(currentScale, Emgu.CV.CvEnum.INTER.CV_INTER_LINEAR);
                var faceRect = FindFace(scaledObservedImage, eigen);

                if (faceRect != Rectangle.Empty)
                {
                    scaledObservedImage.ROI = faceRect;
                    var realImage = new Image<Bgr, int>(observedImagePath);
                    realImage = realImage.Resize(currentScale, Emgu.CV.CvEnum.INTER.CV_INTER_NN);
                    realImage.Draw(faceRect, new Bgr(Color.Blue), 1);
                    realImage.Save(@"output\" + Path.GetFileName(observedImagePath));

                    var fast = new FastDetector(5, true);
                    var brief = new BriefDescriptorExtractor();

                    int result = MatchFace(scaledObservedImage, "fast_set", fast, brief, DistanceType.Hamming);
                    Console.WriteLine("{0}  ==> {1}", observedImagePath, result);

                    return;
                }
                else
                {
                    currentScale += 0.025;
                }
            }

            Console.WriteLine("Match not found");
            
        }

        static Rectangle FindFace(Image<Gray, Byte> observedImage, EigenObjectRecognizer eigen)
        {
            //Equalize histogram
            var c1 = observedImage[0];
            c1._EqualizeHist();
            observedImage[0] = c1;

            var scanRect = new Rectangle(0, 0, 130, 180);

            var imageHeight = observedImage.Height;
            var imageWidth = observedImage.Width;

            int minX, maxX, minY, maxY;
            minX = minY = Int32.MaxValue;
            maxX = maxY = 0;

            bool matchFound = false;

            var foundSpots = new List<KeyValuePair<EigenObjectRecognizer.RecognitionResult, KeyValuePair<int, int>>>();

            while (scanRect.Y < imageHeight - 180)
            {
                while (scanRect.X < imageWidth - 130)
                {
                    observedImage.ROI = scanRect;

                    //Try to recognize it
                    var match = eigen.Recognize(observedImage);

                    if (match != null)
                    {
                        if (scanRect.X < minX) minX = scanRect.X;
                        if (scanRect.Y < minY) minY = scanRect.Y;
                        if (scanRect.X > maxX) maxX = scanRect.X;
                        if (scanRect.Y > maxY) maxY = scanRect.Y;
                        matchFound = true;
                        foundSpots.Add(new KeyValuePair<EigenObjectRecognizer.RecognitionResult, KeyValuePair<int, int>>(match, new KeyValuePair<int, int>(scanRect.X, scanRect.Y)));
                    }

                    scanRect.X += 5;
                }

                scanRect.Y += 5;
                scanRect.X = 0;
            }


            var hehe = foundSpots.OrderBy(s => s.Key.Distance).ToArray();

            if (matchFound)
            {
                var faceRect = new Rectangle(minX, minY, maxX + 130 - minX, maxY + 180 - minY);
                if (faceRect.Width < 140) return faceRect;
                //return faceRect;
            }

            return Rectangle.Empty;
        }

        static List<string> GetPhotosForSubject(int subjectId, string folder)
        {
            var subjectPhotos = new List<string>();
            foreach (var file in Directory.GetFiles(folder, subjectId.ToString() + "_*.bmp"))
            {
                subjectPhotos.Add(file);
            }

            return subjectPhotos;
        }

        static int MatchFace<TDepth>(Image<Gray, byte> imageToMatch, string fastDataset, IKeyPointDetector kp, IDescriptorExtractor<Gray, TDepth> de, DistanceType dt) where TDepth : struct
        {

            var observedKeypoints = kp.DetectKeyPointsRaw(imageToMatch, null);
            var observedDescriptors = de.ComputeDescriptorsRaw(imageToMatch, null, observedKeypoints);

            var dictionary = new Dictionary<int, Person>();

            foreach (var file in Directory.GetFiles(fastDataset, "*.bmp"))
            {
                var subjectId = Convert.ToInt32(Path.GetFileName(file).Split('_')[0]);
                var modelImage = new Image<Gray, byte>(file);

                var modelKeypoints = kp.DetectKeyPointsRaw(modelImage, null);
                var modelDescriptors = de.ComputeDescriptorsRaw(modelImage, null, modelKeypoints);

                Matrix<int> indices;
                Matrix<byte> mask;

                //Try to match the features
                BruteForceMatcher<TDepth> matcher = new BruteForceMatcher<TDepth>(dt);
                matcher.Add(modelDescriptors);

                indices = new Matrix<int>(observedDescriptors.Rows, 5);
                using (Matrix<float> dist = new Matrix<float>(observedDescriptors.Rows, 5))
                {
                    matcher.KnnMatch(observedDescriptors, indices, dist, 5, null);
                    mask = new Matrix<byte>(dist.Rows, 1);
                    mask.SetValue(255);
                    Features2DToolbox.VoteForUniqueness(dist, 0.8, mask);
                }

                int numOfKeypoints = CvInvoke.cvCountNonZero(mask);
                if (numOfKeypoints > 0)
                {
                    numOfKeypoints = Features2DToolbox.VoteForSizeAndOrientation(modelKeypoints, observedKeypoints, indices, mask, 1.5, 10);
                }

                if (!dictionary.ContainsKey(subjectId))
                {
                    dictionary.Add(subjectId, new Person(subjectId));
                }

                dictionary[subjectId].AddComparison((int)(numOfKeypoints));
            }

            return dictionary.Aggregate((l, r) => l.Value.AverageCommonKeypoints > r.Value.AverageCommonKeypoints ? l : r).Key;

        }

        static Image<Gray, Single> ComputeCorrelation(Single[] w, Image<Gray, Single>[] u)
        {
            var result = new Image<Gray, Single>(u[0].Width, u[0].Height, new Gray(0));

            for (int i = 0; i < w.Length; ++i)
            {
                result = result.AddWeighted(u[i], 1.0f, w[0], 0);
            }

            return result;
        }
    }


}
