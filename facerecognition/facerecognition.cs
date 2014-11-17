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
using Emgu.CV.GPU;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Util;
using Emgu.CV.Flann;

using ModelAnalysis;

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
        public const int CROP_WIDTH = 0;

        /// <summary>
        /// Amount of pixel to chop off the top and bottom of the image
        /// </summary>
        public const int CROP_HEIGHT_BOTTOM = 0;

        public const int CROP_HEIGHT_TOP = 0;

        public const float IMAGE_SCALING_FACTOR = 0.65f;

        //public const int POINT_RESPONSE_LOWER_SURF = 3000;
        //public const int HESSIAN_TRESH = 300;

        /// <summary>
        /// The entry point of the code
        /// </summary>
        /// <param name="args">The command line argument(s)</param>
        static void Main(string[] args)
        {
            //if (args.Length == 0)
            //{
            //    Console.Out.WriteLine("You must use this program with a picture in the following form:");
            //    Console.Out.WriteLine("facerecognition subjectID_imageID.gif");
            //}
            //else
            //{
            //    if (!File.Exists(args[0]))
            //    {
            //        Console.Out.Write("The specified file cannot be found in the current folder");
            //        return;
            //    }

            //    Console.Out.WriteLine(IdentifyFaceWithDataset(args[0], trainingDataset));
            //}

            //Tests.testAverageKeypointFastOnRealPhotos();
            Tests.testAverageKeypointFast();
        }

        /*
        public static int IdentifyAverageCommonKeypointFast(string filePath, bool shouldExtractDatabase = true)
        {
            string folderPath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            //var dbFiles = Directory.GetFiles(Path.Combine(folderPath, Program.trainingDataset), "*.gif");
            var dictionary = new Dictionary<int, Person>();

            if (shouldExtractDatabase)
            {
           //     database = ExtractDatabase(folderPath);
            }

            VectorOfKeyPoint unknownKeyPoints;
            Matrix<Byte> unknownDescriptors;

            using (FastDetector fastCPU = new FastDetector(FAST_TRESHOLD_PROCESSING, NON_MAXIMAL_SUPRESSION))
            using (var descriptor = new BriefDescriptorExtractor())
            using (var unknownImage = new Image<Gray, byte>(filePath))
            {
                unknownKeyPoints = fastCPU.DetectKeyPointsRaw(unknownImage, null);
                unknownDescriptors = descriptor.ComputeDescriptorsRaw(unknownImage, null, unknownKeyPoints);
            }
            VectorOfKeyPoint dbKeyPoints;
            foreach (var subject in ModelAnalysisDataSerializer.GetModelAnalyses(Path.Combine(folderPath, "Database")))
            {
                foreach(var photo in subject.photoAnalyses)
                {
                    dbKeyPoints = new VectorOfKeyPoint();
                    dbKeyPoints.Push(photo.keypoints);
                    int computedKeypoints = facerecognition.GetCommonKeypointsFast(unknownDescriptors, unknownKeyPoints, photo.descriptors, dbKeyPoints);

                    //TODO: handle possible error from parsing
                    if (!dictionary.ContainsKey(subject.subjectId))
                    {
                        dictionary.Add(subject.subjectId, new Person(subject.subjectId));
                    }

                    var person = new Person(0);
                    if (!dictionary.TryGetValue(subject.subjectId, out person))
                    {
                        throw new InvalidOperationException("something went wrong with the dictionary");
                    }

                    person.AddComparison(computedKeypoints);
                }
            }

            return dictionary.Aggregate((l, r) => l.Value.AverageCommonKeypoints > r.Value.AverageCommonKeypoints ? l : r).Key;
        }*/

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
        /// Give a cropped, Laplace'd and scalled image of a given picture
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
                var unknownImage = CropAndScalePicture(filePath);

                unknownKeyPoints = fastCPU.DetectKeyPointsRaw(unknownImage, null);
                unknownDescriptors = descriptor.ComputeDescriptorsRaw(unknownImage, null, unknownKeyPoints);

                fastCPU = new FastDetector(FAST_TRESHOLD_ACQUIRING, NON_MAXIMAL_SUPRESSION);
                foreach (var dbFile in dbFiles)
                {

                    // TODO ::   ERASE THIS WHEN DONE TESTINGGGGGGGGGGGGGGGGGGGGGGGGGGGG@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    if (dbFile.Equals(filePath))
                    {
                        continue;
                    }

                    var dbImage = CropAndScalePicture(dbFile);


                    VectorOfKeyPoint dbKeyPoints = new VectorOfKeyPoint();

                    //var dbKeypointArray = fastCPU.DetectKeyPointsRaw(dbImage, null).ToArray().Where(k => k.Response < POINT_RESPONSE_UPPER && k.Response > POINT_RESPONSE_LOWER).ToArray();
                    //dbKeyPoints.Push(dbKeypointArray);
                    dbKeyPoints = fastCPU.DetectKeyPointsRaw(dbImage, null);

                    Matrix<Byte> dbDescriptors = descriptor.ComputeDescriptorsRaw(dbImage, null, dbKeyPoints);

                    int computedKeypoints = facerecognition.GetCommonKeypointsFast(unknownDescriptors, unknownKeyPoints, dbDescriptors, dbKeyPoints);

                    //TODO: handle possible error from parsing
                    int subjectId = Convert.ToInt32(Path.GetFileName(dbFile.Split('_')[0]));

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




        /// <summary>
        /// Identifies the face on the file given, using the dataset at the given path
        /// </summary>
        /// <param name="filePath"></param>
        /// <param name="datasetPath"></param>
        /// <returns></returns>
        /*public static int IdentifyFaceWithDatasetSurf(string filePath, string datasetPath)
        {
            var dbFiles = Directory.GetFiles(datasetPath, "*.gif");
            var dictionary = new Dictionary<int, Person>();
            VectorOfKeyPoint unknownKeyPoints = new VectorOfKeyPoint();
            Matrix<float> unknownDescriptors;
            SURFDetector surfCPU = new SURFDetector(HESSIAN_TRESH, NON_MAXIMAL_SUPRESSION);
            using (var unknownImage = new Image<Gray, byte>(filePath))
            {
                var keypointArray = surfCPU.DetectKeyPointsRaw(unknownImage, null).ToArray().Where(k => k.Response > POINT_RESPONSE_LOWER_SURF).ToArray();
                unknownKeyPoints.Push(keypointArray);
                unknownDescriptors = surfCPU.ComputeDescriptorsRaw(unknownImage, null, unknownKeyPoints);


                IList<IndecesMapping> imap;

                // compute descriptors for each image
                var dbDescsList = ComputeMultipleDescriptors(dbFiles, out imap);

                // concatenate all DB images descriptors into single Matrix
                Matrix<float> dbDescs = ConcatDescriptors(dbDescsList);

                // compute descriptors for the query image
                Matrix<float> queryDescriptors = ComputeSingleDescriptors(filePath);

                FindMatches(dbDescs, queryDescriptors, ref imap);

                var max = imap.OrderByDescending(i => i.IndexEnd).FirstOrDefault();
                return 3;


                /*foreach (var dbFile in dbFiles)
                {

                    // TODO ::   ERASE THIS WHEN DONE TESTINGGGGGGGGGGGGGGGGGGGGGGGGGGGG@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    if (dbFile.Equals(filePath))
                    {
                        continue;
                    }

                    var dbImage = new Image<Gray, byte>(dbFile);


                    VectorOfKeyPoint dbKeyPoints = new VectorOfKeyPoint();

                    var dbKeypointArray = surfCPU.DetectKeyPointsRaw(dbImage, null).ToArray().Where(k => k.Response > POINT_RESPONSE_LOWER_SURF).ToArray();
                    dbKeyPoints.Push(dbKeypointArray);

                    Matrix<float> dbDescriptors = surfCPU.ComputeDescriptorsRaw(dbImage, null, dbKeyPoints);


                    IList<IndecesMapping> imap = new List<IndecesMapping>();

                    GetMatches(unknownDescriptors, unknownKeyPoints, dbDescriptors, dbKeyPoints, ref imap);
                    int computedKeypoints = imap.Count;//facerecognition.GetCommonKeypointsSurf(unknownDescriptors, unknownKeyPoints, dbDescriptors, dbKeyPoints);

                    //TODO: handle possible error from parsing
                    int subjectId = Convert.ToInt32(Path.GetFileName(dbFile.Split('_')[0]));

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
        }*/

        /*public static Dictionary<int, ICollection<PhotoAnalysisData>> ExtractDatabase(string currentFolder)
        {
            string databaseFolder = Path.Combine(currentFolder, "Database");
            var dictionary = new Dictionary<int, ICollection<PhotoAnalysisData>>();
            //Deserialize objects
            foreach (var modelData in ModelAnalysisDataSerializer.GetModelAnalyses(databaseFolder))
            {
                dictionary.Add(modelData.subjectId, modelData.photoAnalyses);
            }

            return dictionary;
        }*/

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
            FastDetector fastCPU = new FastDetector(FAST_TRESHOLD_PROCESSING, NON_MAXIMAL_SUPRESSION);
            Matrix<int> indices;

            BriefDescriptorExtractor descriptor = new BriefDescriptorExtractor();

            Matrix<byte> mask;

            // extract features from the observed image
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
            if (numOfKeypoints >= 4)
            {
                return Features2DToolbox.VoteForSizeAndOrientation(unknownImageKeyPoints, dbImageKeyPoints, indices, mask, SCALE_INCREMENT, ROTATION_BINS);
            }

            return numOfKeypoints;
        }

        /*
        public static int GetCommonKeypointsSurf(Matrix<float> unknownDescriptors, VectorOfKeyPoint unknownImageKeyPoints, Matrix<float> dbDescriptors, VectorOfKeyPoint dbImageKeyPoints, ref IList<IndecesMapping> imap)
        {
            SURFDetector fastCPU = new SURFDetector(HESSIAN_TRESH, NON_MAXIMAL_SUPRESSION);
            Matrix<int> indices = new Matrix<int>(dbDescriptors.Rows, 2);
            var dists = new Matrix<float>(dbDescriptors.Rows, 2);

            BriefDescriptorExtractor descriptor = new BriefDescriptorExtractor();

            Matrix<byte> mask;

            // extract features from the observed image

            var flannIndex = new Index(unknownDescriptors, 4);
            flannIndex.KnnSearch(dbDescriptors, indices, dists, 2, 24);
            for (int i = 0; i < indices.Rows; i++)
            {
                // filter out all inadequate pairs based on distance between pairs
                if (dists.Data[i, 0] < (0.6 * dists.Data[i, 1]))
                {
                    // find image from the db to which current descriptor range belongs and increment similarity value.
                    // in the actual implementation this should be done differently as it's not very efficient for large image collections.
                    foreach (var img in imap)
                    {
                        if (img.IndexStart <= i && img.IndexEnd >= i)
                        {
                            img.Similarity++;
                            break;
                        }
                    }
                }
            }

            BruteForceMatcher<float> matcher = new BruteForceMatcher<float>(BFM_OPTION);
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
            if (numOfKeypoints >= 4)
            {
                return Features2DToolbox.VoteForSizeAndOrientation(unknownImageKeyPoints, dbImageKeyPoints, indices, mask, SCALE_INCREMENT, ROTATION_BINS);
            }

            return numOfKeypoints;
        }*/
    }
}
