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
using ModelAnalysis;

namespace facerecognition
{
    public class Program
    {
        public const int HESSIAN_TRESHOLD = 750;

        public const int FAST_TRESHOLD_PROCESSING = 7;

        public const int FAST_TRESHOLD_ACQUIRING = 2;

        public const int KNN_MATCH_CONSTANT = 2;

        public const double UNIQUENESS_THRESHOLD = 0.8;

        public const double SCALE_INCREMENT = 1.5;

        public const int ROTATION_BINS = 10;

        public const bool NON_MAXIMAL_SUPRESSION = true;

        public const string trainingDataset = "training dataset";

        public const DistanceType BFM_OPTION = DistanceType.Hamming;

        public static Dictionary<int, ICollection<PhotoAnalysisData<byte>>> database = new Dictionary<int,ICollection<PhotoAnalysisData<byte>>>();
        
        static void Main(string[] args)
        {
            //if (args.Length == 0)
            //{
            //    Console.Out.WriteLine("You must use this program with a picture in the following form:");
            //    Console.Out.WriteLine("facerecognition subjectID_imageID.gif");
            //}
            //else
            //{
            //    string filePath = GetFilePath(args[0]);
            //    Console.Out.WriteLine(IdentifyAverageCommonKeypointFast(filePath));
            //}

            Tests.testAverageKeypointFast();
        }

        public static int IdentifyAverageCommonKeypoint(string filePath)
        {
            string folderPath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            var dbFiles = Directory.GetFiles(Path.Combine(folderPath, Program.trainingDataset), "*.gif");
            var dictionary = new Dictionary<int, Person>();

            using (SURFDetector surfCPU = new SURFDetector(HESSIAN_TRESHOLD, false))
            {
                var unknownImage = new Image<Gray, byte>(filePath);

                VectorOfKeyPoint unknownKeyPoints = surfCPU.DetectKeyPointsRaw(unknownImage, null);
                Matrix<float> unknownDescriptors = surfCPU.ComputeDescriptorsRaw(unknownImage, null, unknownKeyPoints);

                foreach (var dbFile in dbFiles)
                {
                    var dbImage = new Image<Gray, byte>(dbFile);



                    VectorOfKeyPoint dbKeyPoints = surfCPU.DetectKeyPointsRaw(dbImage, null);
                    Matrix<float> dbDescriptors = surfCPU.ComputeDescriptorsRaw(dbImage, null, dbKeyPoints);

                    int computedKeypoints = Program.GetCommonKeypoints(unknownDescriptors, unknownKeyPoints, dbDescriptors, dbKeyPoints);

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
            foreach (var subject in ModelAnalysisDataSerializer.GetModelAnalyses<byte>(Path.Combine(folderPath, "Database")))
            {
                foreach(var photo in subject.photoAnalyses)
                {
                    dbKeyPoints = new VectorOfKeyPoint();
                    dbKeyPoints.Push(photo.keypoints);
                    int computedKeypoints = Program.GetCommonKeypointsFast(unknownDescriptors, unknownKeyPoints, photo.descriptors, dbKeyPoints);

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
        }

        public static int IdentifyAverageCommonKeypointFastWithoutDb(string filePath)
        {
            string folderPath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            var dbFiles = Directory.GetFiles(Path.Combine(folderPath, Program.trainingDataset), "*.gif");
            var dictionary = new Dictionary<int, Person>();
            VectorOfKeyPoint unknownKeyPoints;
            Matrix<Byte> unknownDescriptors;
            FastDetector fastCPU = new FastDetector(FAST_TRESHOLD_PROCESSING, NON_MAXIMAL_SUPRESSION);
            using (var descriptor = new BriefDescriptorExtractor())
            using (var unknownImage = new Image<Gray, byte>(filePath))
            {
                unknownKeyPoints = fastCPU.DetectKeyPointsRaw(unknownImage, null);
                unknownDescriptors = descriptor.ComputeDescriptorsRaw(unknownImage, null, unknownKeyPoints);

                fastCPU = new FastDetector(FAST_TRESHOLD_ACQUIRING, NON_MAXIMAL_SUPRESSION);
                foreach (var dbFile in dbFiles)
                {
                    var dbImage = new Image<Gray, byte>(dbFile);


                    VectorOfKeyPoint dbKeyPoints = fastCPU.DetectKeyPointsRaw(dbImage, null);
                    Matrix<Byte> dbDescriptors = descriptor.ComputeDescriptorsRaw(dbImage, null, dbKeyPoints);

                    int computedKeypoints = Program.GetCommonKeypointsFast(unknownDescriptors, unknownKeyPoints, dbDescriptors, dbKeyPoints);

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

        public static Dictionary<int, ICollection<PhotoAnalysisData<byte>>> ExtractDatabase(string currentFolder)
        {
            string databaseFolder = Path.Combine(currentFolder, "Database");
            var dictionary = new Dictionary<int, ICollection<PhotoAnalysisData<byte>>>();
            //Deserialize objects
            foreach (var modelData in ModelAnalysisDataSerializer.GetModelAnalyses<byte>(databaseFolder))
            {
                dictionary.Add(modelData.subjectId, modelData.photoAnalyses);
            }

            return dictionary;
        }
        
        static string GetFilePath(string arg)
        {
            string folderPath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            string completeFilePath = Path.Combine(folderPath, arg);
            if (!File.Exists(completeFilePath))
            {
                Console.Out.Write("The specified file cannot be found in the current folder");
                throw new FileNotFoundException("Could not find the file", arg);
            }

            return completeFilePath;
        }


        public static int GetCommonKeypointsOriginal(Image<Gray, Byte> imageToIdentify, Image<Gray, byte> databaseImage)
        {
            SURFDetector surfCPU = new SURFDetector(HESSIAN_TRESHOLD, false);
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            Matrix<int> indices;

            Matrix<byte> mask;
            //extract features from the object image
            modelKeyPoints = surfCPU.DetectKeyPointsRaw(imageToIdentify, null);
            Matrix<float> modelDescriptors = surfCPU.ComputeDescriptorsRaw(imageToIdentify, null, modelKeyPoints);

            // extract features from the observed image
            observedKeyPoints = surfCPU.DetectKeyPointsRaw(databaseImage, null);
            Matrix<float> observedDescriptors = surfCPU.ComputeDescriptorsRaw(databaseImage, null, observedKeyPoints);
            BruteForceMatcher<float> matcher = new BruteForceMatcher<float>(BFM_OPTION);
            matcher.Add(modelDescriptors);

            indices = new Matrix<int>(observedDescriptors.Rows, KNN_MATCH_CONSTANT);
            using (Matrix<float> dist = new Matrix<float>(observedDescriptors.Rows, KNN_MATCH_CONSTANT))
            {
                matcher.KnnMatch(observedDescriptors, indices, dist, KNN_MATCH_CONSTANT, null);
                mask = new Matrix<byte>(dist.Rows, 1);
                mask.SetValue(255);
                Features2DToolbox.VoteForUniqueness(dist, UNIQUENESS_THRESHOLD, mask);
            }

            int nonZeroCount = CvInvoke.cvCountNonZero(mask);
            if (nonZeroCount >= 4)
            {
                nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, indices, mask, SCALE_INCREMENT, ROTATION_BINS);
            }

            return nonZeroCount;
        }

        public static int GetCommonKeypointsFast(Matrix<Byte> unknownDescriptors, VectorOfKeyPoint unknownImageKeyPoints, Matrix<Byte> dbDescriptors, VectorOfKeyPoint dbImageKeyPoints)
        {
            FastDetector fastCPU = new FastDetector(FAST_TRESHOLD_PROCESSING, NON_MAXIMAL_SUPRESSION);
            Matrix<int> indices;

            BriefDescriptorExtractor descriptor = new BriefDescriptorExtractor();

            Matrix<byte> mask;

            // extract features from the observed image
            BruteForceMatcher<Byte> matcher = new BruteForceMatcher<Byte>(BFM_OPTION);
            matcher.Add(unknownDescriptors);

            indices = new Matrix<int>(dbDescriptors.Rows, KNN_MATCH_CONSTANT);
            using (Matrix<float> dist = new Matrix<float>(dbDescriptors.Rows, KNN_MATCH_CONSTANT))
            {
                matcher.KnnMatch(dbDescriptors, indices, dist, KNN_MATCH_CONSTANT, null);
                mask = new Matrix<byte>(dist.Rows, 1);
                mask.SetValue(255);
                Features2DToolbox.VoteForUniqueness(dist, UNIQUENESS_THRESHOLD, mask);
            }

            return Features2DToolbox.VoteForSizeAndOrientation(unknownImageKeyPoints, dbImageKeyPoints, indices, mask, SCALE_INCREMENT, ROTATION_BINS);
        }

      public static int GetCommonKeypoints(Matrix<float> unknownImageDescriptor, VectorOfKeyPoint unknownImageKeyPoints, Matrix<float> dbImageDescriptor, VectorOfKeyPoint dbImageKeyPoints)
      {
          SURFDetector surfCPU = new SURFDetector(HESSIAN_TRESHOLD, false);
          Matrix<int> indices;

          Matrix<byte> mask;
          int k = 2;
          double uniquenessThreshold = 0.8;

          BruteForceMatcher<float> matcher = new BruteForceMatcher<float>(DistanceType.L2);
          matcher.Add(unknownImageDescriptor);

          indices = new Matrix<int>(dbImageDescriptor.Rows, k);
          using (Matrix<float> dist = new Matrix<float>(dbImageDescriptor.Rows, k))
          {
              matcher.KnnMatch(dbImageDescriptor, indices, dist, k, null);
              mask = new Matrix<byte>(dist.Rows, 1);
              mask.SetValue(255);
              Features2DToolbox.VoteForUniqueness(dist, uniquenessThreshold, mask);
          }

          int nonZeroCount = CvInvoke.cvCountNonZero(mask);
          if (nonZeroCount >= 4)
          {
              nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(unknownImageKeyPoints, dbImageKeyPoints, indices, mask, 1.5, ROTATION_BINS);
          }

          return nonZeroCount;
      }
    }
}
