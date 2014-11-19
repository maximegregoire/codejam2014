/// <copyright file="facerecognition.cs">
/// Copyright (c) 2014 All Rights Reserved
/// </copyright>
/// <author>Maxime Grégoire</author>
/// <author>Kevin Cadieux</author>
/// <summary>
/// Class responsible for handling the facial recognition and all of its processes
/// </summary>

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Drawing;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Windows.Forms;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Util;
using Database;

namespace facerecognition
{
    /// <summary>
    /// Class responsible for handling the recognition and all of its processes
    /// </summary>
    public class facerecognition
    {
        /// <summary>
        /// The FAST detector's threshold for processing data (during comparison of pictures)
        /// </summary>
        public const int FAST_TRESHOLD_PROCESSING = 7;

        /// <summary>
        /// The FAST detector's threshold for acquiring data (during acquisition of dataset keypoints)
        /// </summary>
        public const int FAST_TRESHOLD_ACQUIRING = 2;

        /// <summary>
        /// The KNN match constant (to find best matches with detectors)
        /// </summary>
        public const int KNN_MATCH_CONSTANT = 10;

        /// <summary>
        /// The uniqueness threshold (to determine the unique keypoints)
        /// </summary>
        public const double UNIQUENESS_THRESHOLD = 0.8;

        /// <summary>
        /// The scale increment of the non-zero keypoints
        /// </summary>
        public const double SCALE_INCREMENT = 1.5;

        /// <summary>
        /// The rotation bins of the non-zero keypoints
        /// </summary>
        public const int ROTATION_BINS = 10;

        /// <summary>
        /// Allows whether or not the supression of non-maximal points is allowed
        /// </summary>
        public const bool NON_MAXIMAL_SUPRESSION = true;

        /// <summary>
        /// The name of the training dataset folder
        /// </summary>
        public const string trainingDataset = "training dataset";

        /// <summary>
        /// The configuration for the distance type of the brute-force matcher
        /// </summary>
        public const DistanceType BFM_OPTION = DistanceType.Hamming;

        /// <summary>
        /// Amount of pixel to chop off both sides of the image
        /// </summary>
        public const int CROP_WIDTH = 30;

        /// <summary>
        /// Amount of pixel to chop off of the bottom of the image
        /// </summary>
        public const int CROP_HEIGHT_BOTTOM = 30;

        /// <summary>
        /// Amount of pixel to chop off of the top of the image
        /// </summary>
        public const int CROP_HEIGHT_TOP = 30;

        /// <summary>
        /// Amount by which the image will be scaled (1 = original size, 0.5 = 50%, 2 = 200%, etc..)
        /// </summary>
        public const float IMAGE_SCALING_FACTOR = 0.55f;

        /// <summary>
        /// Threshold for the amout of significant keypoint matches
        /// </summary>
        public const int SIGNIFICANT_KEYPOINT_THRESHOLD = 4;

        static Dictionary<int, SubjectModelDataCollection<byte>> db;

        /// <summary>
        /// The entry point of the code
        /// </summary>
        /// <param name="args">The command line argument(s)</param>
        static void Main(string[] args)
        {
           /* 
            if (args.Length == 0)
            {
                Console.Out.WriteLine("You must use this program with a picture in the following form:");
                Console.Out.WriteLine("facerecognition subjectID_imageID.gif");
            }
            else
            {
                if (!File.Exists(args[0]))
                {
                    Console.Out.WriteLine("The specified file cannot be found in the current folder.");
                    return;
                }

                if (!IsImage(args[0]))
                {
                    Console.Out.WriteLine("The specified file is not an image.");
                    return;
                }

                Console.WriteLine(IdentifyFaceWithDataset(args[0], trainingDataset));
            }*/

            db = Serializer.ReadDatabase<byte>("database_v2");

            //PreprocessImages("photos", "DatabaseReal");
            //Tests.testRecognitionYale();
            if (args.Length > 0 && args[0] == "old") Tests.testRecognitionOnRealPhotos(false);
            else Tests.testRecognitionOnRealPhotos(true);
            
        }


        /// <summary>
        /// Checks that a given file is an image
        /// </summary>
        /// <param name="filePath">The path of the file</param>
        /// <returns>True if it's an image, false otherwise</returns>
        public static bool IsImage(string filePath)
        {
            return filePath.EndsWith(".gif") || filePath.EndsWith(".bmp") || filePath.EndsWith(".png") ||
                filePath.EndsWith(".jpg") || filePath.EndsWith(".GIF") || filePath.EndsWith(".BMP") || 
                filePath.EndsWith(".PNG") || filePath.EndsWith(".JPG");
        }

        /// <summary>
        /// Give a cropped and scalled image of a given picture
        /// </summary>
        /// <param name="picturePath">The path of the picture</param>
        /// <returns>The cropped and scalled image</returns>
        public static Image<Gray, byte> CropAndScalePicture(string picturePath)
        {
            Bitmap myBitmap = new Bitmap(picturePath);
            Rectangle cloneRect = new Rectangle(CROP_WIDTH, CROP_HEIGHT_BOTTOM, myBitmap.Width - (2 * CROP_WIDTH), myBitmap.Height - CROP_HEIGHT_BOTTOM - CROP_HEIGHT_TOP);
            Bitmap cloneBitmap = myBitmap.Clone(cloneRect, myBitmap.PixelFormat);
            return new Image<Gray, byte>(new Bitmap(cloneBitmap, new Size((int)(cloneBitmap.Width * IMAGE_SCALING_FACTOR), (int)(cloneBitmap.Height * IMAGE_SCALING_FACTOR)))); ;
        }

        /// <summary>
        /// Give a cropped, Laplace'd and scaled image of a given picture
        /// </summary>
        /// <param name="picturePath">The path of the picture</param>
        /// <returns>The transformed image</returns>
        public static Image<Gray, byte> TransformPicture(string picturePath)
        {
            Bitmap myBitmap = new Bitmap(picturePath);
            Rectangle cloneRect = new Rectangle(CROP_WIDTH, CROP_HEIGHT_BOTTOM, myBitmap.Width - (2 * CROP_WIDTH), myBitmap.Height - CROP_HEIGHT_BOTTOM - CROP_HEIGHT_TOP);
            Bitmap cloneBitmap = myBitmap.Clone(cloneRect, myBitmap.PixelFormat);

            var bmp = new Image<Gray, byte>(new Bitmap(cloneBitmap, new Size((int)(cloneBitmap.Width * IMAGE_SCALING_FACTOR), (int)(cloneBitmap.Height * IMAGE_SCALING_FACTOR))));
            var lap = bmp.Laplace(1);
            var aa = lap.ToBitmap();
            bmp = new Image<Gray, byte>(aa);
            return bmp;
        }            

        /// <summary>
        /// Identifies the face on the file given, using the dataset at the given path
        /// </summary>
        /// <param name="filePath">full path of the picture to identify</param>
        /// <param name="datasetPath">full path of the training dataset</param>
        /// <returns>The ID of the face on the picture</returns>
        public static int IdentifyFaceWithDataset(string filePath, string datasetPath)
        {
            var dbFiles = Directory.GetFiles(datasetPath).Where(f => f.EndsWith(".gif") || f.EndsWith(".GIF") || f.EndsWith(".bmp") || f.EndsWith(".BMP") || f.EndsWith(".jpg") || f.EndsWith(".JPG") || f.EndsWith(".PNG") || f.EndsWith(".png"));
            var dictionary = new Dictionary<int, Person>();
            VectorOfKeyPoint unknownKeyPoints = new VectorOfKeyPoint();
            Matrix<Byte> unknownDescriptors;
            FastDetector fastCPU = new FastDetector(FAST_TRESHOLD_PROCESSING, NON_MAXIMAL_SUPRESSION);
            using (var descriptor = new BriefDescriptorExtractor())
            {
                var unknownImage = TransformPicture(filePath);

                unknownKeyPoints = fastCPU.DetectKeyPointsRaw(unknownImage, null);
                unknownDescriptors = descriptor.ComputeDescriptorsRaw(unknownImage, null, unknownKeyPoints);

                fastCPU = new FastDetector(FAST_TRESHOLD_ACQUIRING, NON_MAXIMAL_SUPRESSION);
                foreach (var dbFile in dbFiles)
                {

                    //TODO: Remove
                    /*
                    if (Path.GetFileName(dbFile) == Path.GetFileName(filePath))
                    {
                        continue;
                    }*/

                    var dbImage = TransformPicture(dbFile);
                    //var dbImage = new Image<Gray, byte>(dbFile);

                    VectorOfKeyPoint dbKeyPoints = fastCPU.DetectKeyPointsRaw(dbImage, null);
                    Matrix<Byte> dbDescriptors = descriptor.ComputeDescriptorsRaw(dbImage, null, dbKeyPoints);

                    int computedKeypoints = facerecognition.GetCommonKeypointsFast(unknownDescriptors, unknownKeyPoints, dbDescriptors, dbKeyPoints);

                    int subjectId = Convert.ToInt32(Path.GetFileName(dbFile).Split('_')[0]);

                    if (!dictionary.ContainsKey(subjectId))
                    {
                        dictionary.Add(subjectId, new Person(subjectId));
                    }

                    var person = new Person(0);
                    if (!dictionary.TryGetValue(subjectId, out person))
                    {
                        throw new InvalidOperationException("something went wrong with the dictionary");
                    }

                    person.AddComparison(computedKeypoints);

                }

                return dictionary.Aggregate((l, r) => l.Value.AverageCommonKeypoints > r.Value.AverageCommonKeypoints ? l : r).Key;
            }
        }

        public static int IdentifyFaceWithDatabase(string filePath)
        {
            var observedImage = new Image<Gray, byte>(filePath);

            var dictionary = new Dictionary<int, Person>();

            //var orb = new ORBDetector(500);
            //var surf = new SURFDetector(500, true);
            //var brisk = new Brisk(1, 3, 1.0f);
            var fast = new FastDetector(1, true);
            //var freak = new Freak(true, true, 22.0f, 3);
            var brief = new BriefDescriptorExtractor();

            var observedKeypoints = fast.DetectKeyPointsRaw(observedImage, null);
            var observedDescriptors = brief.ComputeDescriptorsRaw(observedImage, null, observedKeypoints);

            foreach (var subjectId in db.Keys)
            {
                var models = db[subjectId].models;

                foreach (var model in models)
                {
                    Matrix<int> indices;
                    Matrix<byte> mask;

                    //Try to match the features
                    BruteForceMatcher<byte> matcher = new BruteForceMatcher<byte>(DistanceType.Hamming);
                    matcher.Add(model.descriptors);

                    indices = new Matrix<int>(observedDescriptors.Rows, 10);
                    using (Matrix<float> dist = new Matrix<float>(observedDescriptors.Rows, 10))
                    {
                        matcher.KnnMatch(observedDescriptors, indices, dist, 10, null);
                        mask = new Matrix<byte>(dist.Rows, 1);
                        mask.SetValue(255);
                        Features2DToolbox.VoteForUniqueness(dist, 0.6, mask);
                    }

                    var kpModel = new VectorOfKeyPoint();
                    kpModel.Push(model.keypoints);
                    int numOfKeypoints = CvInvoke.cvCountNonZero(mask);
                    if (numOfKeypoints > 0)
                    {
                        numOfKeypoints = Features2DToolbox.VoteForSizeAndOrientation(kpModel, observedKeypoints, indices, mask, 1.5, 20);
                    }

                    if (!dictionary.ContainsKey(subjectId))
                    {
                        dictionary.Add(subjectId, new Person(subjectId));
                    }

                    dictionary[subjectId].AddComparison(numOfKeypoints);
                }
            }

            return dictionary.Aggregate((l, r) => l.Value.AverageCommonKeypoints > r.Value.AverageCommonKeypoints ? l : r).Key;
        }

        /// <summary>
        /// Utility function to preprocess all images from training dataset
        /// </summary>
        /// <param name="dbPathIn">full path of training dataset</param>
        /// <param name="dbPathOut">full path of destination preprocessed image folder</param>
        /// <returns>The ID of the face on the picture</returns>
        public static void PreprocessImages(string dbPathIn, string dbPathOut)
        {
            Directory.CreateDirectory(dbPathOut);
            foreach (var file in Directory.GetFiles(dbPathIn).Where(f => f.EndsWith(".gif") || f.EndsWith(".GIF") || f.EndsWith(".bmp") || f.EndsWith(".BMP") || f.EndsWith(".jpg") || f.EndsWith(".JPG") || f.EndsWith(".PNG") || f.EndsWith(".png")))
            {
                var fileName = Path.GetFileName(file);
                var image = TransformPicture(file);

                var newFilePath = Path.Combine(dbPathOut, fileName);
                image.Save(newFilePath);
            }
        }

        /// <summary>
        /// Gets the amount of common key points between two sets of descriptors and key points, using a fast generator
        /// </summary>
        /// <param name="unknownDescriptors">The descriptors of the picture to identify</param>
        /// <param name="unknownImageKeyPoints">The key points of the picture to identify</param>
        /// <param name="dbDescriptors">A dataset's descriptor</param>
        /// <param name="dbImageKeyPoints">A dataset's key points set</param>
        /// <returns>The amount of common key points</returns>
        public static int GetCommonKeypointsFast(Matrix<Byte> unknownDescriptors, VectorOfKeyPoint unknownImageKeyPoints, Matrix<Byte> dbDescriptors, VectorOfKeyPoint dbImageKeyPoints)
        {
            Matrix<int> indices;
            Matrix<byte> mask;

            //Try to match the features
            BruteForceMatcher<Byte> matcher = new BruteForceMatcher<Byte>(BFM_OPTION);
            matcher.Add(unknownDescriptors);

            if (dbDescriptors == null)
            {
                return 0;
            }

            indices = new Matrix<int>(dbDescriptors.Rows, KNN_MATCH_CONSTANT);
            using (Matrix<float> dist = new Matrix<float>(dbDescriptors.Rows, KNN_MATCH_CONSTANT))
            {
                matcher.KnnMatch(dbDescriptors, indices, dist, KNN_MATCH_CONSTANT, null);
                mask = new Matrix<byte>(dist.Rows, 1);
                mask.SetValue(255);
                Features2DToolbox.VoteForUniqueness(dist, UNIQUENESS_THRESHOLD, mask);
            }

            int numOfKeypoints = CvInvoke.cvCountNonZero(mask);
            if (numOfKeypoints >= SIGNIFICANT_KEYPOINT_THRESHOLD)
            {
                return Features2DToolbox.VoteForSizeAndOrientation(unknownImageKeyPoints, dbImageKeyPoints, indices, mask, SCALE_INCREMENT, ROTATION_BINS);
            }

            return numOfKeypoints;
        }
    }
}
