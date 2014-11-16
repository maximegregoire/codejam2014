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

namespace facerecognition
{
    class Program
    {
        public const int HESSIAN_TRESHOLD = 750;

        public const int FAST_TRESHOLD = 2;

        public const bool NON_MAXIMAL_SUPRESSION = true;

        public const string trainingDataset = "training dataset";

        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.Out.WriteLine("You must use this program with a picture in the following form:");
                Console.Out.WriteLine("facerecogntion subjectID_imageID.gif");
            }
            else
            {
                string filePath = GetFilePath(args[0]);
                Console.Out.WriteLine(IdentifyAverageCommonKeypointFast(filePath));
            }
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

        public static int IdentifyAverageCommonKeypointFast(string filePath)
        {
            string folderPath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            var dbFiles = Directory.GetFiles(Path.Combine(folderPath, Program.trainingDataset), "*.gif");
            var dictionary = new Dictionary<int, Person>();

            using (FastDetector fastCPU = new FastDetector(FAST_TRESHOLD, NON_MAXIMAL_SUPRESSION))
            using (var descriptor = new BriefDescriptorExtractor())
            {
                var unknownImage = new Image<Gray, byte>(filePath);

                VectorOfKeyPoint unknownKeyPoints = fastCPU.DetectKeyPointsRaw(unknownImage, null);
                Matrix<Byte> unknownDescriptors = descriptor.ComputeDescriptorsRaw(unknownImage, null, unknownKeyPoints);

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
                    if(!dictionary.TryGetValue(subjectId, out person))
                    {
                        throw new InvalidOperationException("something went wrong with the dictionary");
                    }

                    person.AddComparison(computedKeypoints);
                    
                }

                return dictionary.Aggregate((l, r) => l.Value.AverageCommonKeypoints > r.Value.AverageCommonKeypoints ? l : r).Key;
            }
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
            int k = 2;
            double uniquenessThreshold = 0.8;
            //extract features from the object image
            modelKeyPoints = surfCPU.DetectKeyPointsRaw(imageToIdentify, null);
            Matrix<float> modelDescriptors = surfCPU.ComputeDescriptorsRaw(imageToIdentify, null, modelKeyPoints);

            // extract features from the observed image
            observedKeyPoints = surfCPU.DetectKeyPointsRaw(databaseImage, null);
            Matrix<float> observedDescriptors = surfCPU.ComputeDescriptorsRaw(databaseImage, null, observedKeyPoints);
            BruteForceMatcher<float> matcher = new BruteForceMatcher<float>(DistanceType.L2);
            matcher.Add(modelDescriptors);

            indices = new Matrix<int>(observedDescriptors.Rows, k);
            using (Matrix<float> dist = new Matrix<float>(observedDescriptors.Rows, k))
            {
                matcher.KnnMatch(observedDescriptors, indices, dist, k, null);
                mask = new Matrix<byte>(dist.Rows, 1);
                mask.SetValue(255);
                Features2DToolbox.VoteForUniqueness(dist, uniquenessThreshold, mask);
            }

            int nonZeroCount = CvInvoke.cvCountNonZero(mask);
            if (nonZeroCount >= 4)
            {
                nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, indices, mask, 1.5, 20);
            }

            return nonZeroCount;
        }

        public static int GetCommonKeypointsFast(Matrix<Byte> unknownDescriptors, VectorOfKeyPoint unknownImageKeyPoints, Matrix<Byte> dbDescriptors, VectorOfKeyPoint dbImageKeyPoints)
        {
            FastDetector fastCPU = new FastDetector(FAST_TRESHOLD, NON_MAXIMAL_SUPRESSION);
            Matrix<int> indices;

            BriefDescriptorExtractor descriptor = new BriefDescriptorExtractor();

            Matrix<byte> mask;
            int k = 2;
            double uniquenessThreshold = 0.8;

            // extract features from the observed image
            BruteForceMatcher<Byte> matcher = new BruteForceMatcher<Byte>(DistanceType.L2);
            matcher.Add(unknownDescriptors);

            indices = new Matrix<int>(dbDescriptors.Rows, k);
            using (Matrix<float> dist = new Matrix<float>(dbDescriptors.Rows, k))
            {
                matcher.KnnMatch(dbDescriptors, indices, dist, k, null);
                mask = new Matrix<byte>(dist.Rows, 1);
                mask.SetValue(255);
                Features2DToolbox.VoteForUniqueness(dist, uniquenessThreshold, mask);
            }

            int nonZeroCount = CvInvoke.cvCountNonZero(mask);
            if (nonZeroCount >= 4)
            {
                nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(unknownImageKeyPoints, dbImageKeyPoints, indices, mask, 1.5, 20);
            }

            return nonZeroCount;
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
              nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(unknownImageKeyPoints, dbImageKeyPoints, indices, mask, 1.5, 20);
          }

          return nonZeroCount;
      }
    }
}
