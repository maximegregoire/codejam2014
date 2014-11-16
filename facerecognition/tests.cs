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
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace facerecognition
{
    class Tests
    {

        public static void testFullDatabaseFast()
        {
            int results = 0;
            int error = 0;

            string folderPath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            using (FastDetector fastCPU = new FastDetector(10, true))
            using (BriefDescriptorExtractor descriptor = new BriefDescriptorExtractor())
            {
                foreach (var fileToIdentify in Directory.GetFiles(@"C:\Users\Maxime\Downloads\yalefaces\yalefaces", "*.gif"))
                {

                    var unknownImage = new Image<Gray, byte>(fileToIdentify);
                    int maxKeyPoints = 0;
                    string maxKeyPointsImage = string.Empty;


                    VectorOfKeyPoint unknownKeyPoints = fastCPU.DetectKeyPointsRaw(unknownImage, null);
                    Matrix<Byte> unknownDescriptors = descriptor.ComputeDescriptorsRaw(unknownImage, null, unknownKeyPoints);

                    foreach (var dbFile in Directory.GetFiles(@"C:\Users\Maxime\Downloads\yalefaces\yalefaces", "*.gif").Where(f => f != fileToIdentify))
                    {
                        var dbImage = new Image<Gray, byte>(dbFile);
                        VectorOfKeyPoint dbKeyPoints = fastCPU.DetectKeyPointsRaw(dbImage, null);
                        Matrix<Byte> dbDescriptors = descriptor.ComputeDescriptorsRaw(dbImage, null, dbKeyPoints);

                        int computedKeypoints = Program.GetCommonKeypointsFast(unknownDescriptors, unknownKeyPoints, dbDescriptors, dbKeyPoints);
                        if (computedKeypoints > maxKeyPoints)
                        {
                            maxKeyPoints = computedKeypoints;
                            maxKeyPointsImage = dbFile;
                        }
                    }

                    results++;
                    if (!Path.GetFileName(fileToIdentify).Split('.')[0].Equals(Path.GetFileName(maxKeyPointsImage).Split('.')[0]))
                    {
                        Console.Out.WriteLine("ERROR");
                        error++;
                    }

                    Console.Out.WriteLine("Results for model " + Path.GetFileName(fileToIdentify) + " = " + Path.GetFileName(maxKeyPointsImage));
                }
            }

            Console.Out.WriteLine("\n\n\n RESULTS = " + (100 - (((float)(error) / (float)(results)) * 100)));
        }

        public static void testSubsetFast()
        {
            int results = 0;
            int error = 0;

            string folderPath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            using (FastDetector fastCPU = new FastDetector(10, true))
            using (BriefDescriptorExtractor descriptor = new BriefDescriptorExtractor())
            {
                foreach (var fileToIdentify in Directory.GetFiles(Path.Combine(folderPath, Program.trainingDataset), "*.gif"))
                {
                    var unknownImage = new Image<Gray, byte>(fileToIdentify);
                    int maxKeyPoints = 0;
                    string maxKeyPointsImage = string.Empty;


                    VectorOfKeyPoint unknownKeyPoints = fastCPU.DetectKeyPointsRaw(unknownImage, null);
                    Matrix<Byte> unknownDescriptors = descriptor.ComputeDescriptorsRaw(unknownImage, null, unknownKeyPoints);

                    foreach (var dbFile in Directory.GetFiles(Path.Combine(folderPath, Program.trainingDataset), "*.gif").Where(f => f != fileToIdentify))
                    {
                        var dbImage = new Image<Gray, byte>(dbFile);
                        VectorOfKeyPoint dbKeyPoints = fastCPU.DetectKeyPointsRaw(dbImage, null);
                        Matrix<Byte> dbDescriptors = descriptor.ComputeDescriptorsRaw(dbImage, null, dbKeyPoints);

                        int computedKeypoints = Program.GetCommonKeypointsFast(unknownDescriptors, unknownKeyPoints, dbDescriptors, dbKeyPoints);
                        if (computedKeypoints > maxKeyPoints)
                        {
                            maxKeyPoints = computedKeypoints;
                            maxKeyPointsImage = dbFile;
                        }
                    }



                    results++;
                    if (!Path.GetFileName(fileToIdentify).Split('_')[0].Equals(Path.GetFileName(maxKeyPointsImage).Split('_')[0]))
                    {
                        Console.Out.WriteLine("ERROR");
                        error++;
                    }

                    Console.Out.WriteLine("Results for model " + Path.GetFileName(fileToIdentify) + " = " + Path.GetFileName(maxKeyPointsImage));
                }
            }

            Console.Out.WriteLine("\n\n\n RESULTS = " + (100 - (((float)(error) / (float)(results)) * 100)));
        }

        public static void testSubset()
        {
            int results = 0;
            int error = 0;

            string folderPath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            using (SURFDetector surfCPU = new SURFDetector(500, false))
            {
                foreach (var fileToIdentify in Directory.GetFiles(Path.Combine(folderPath, Program.trainingDataset), "*.gif"))
                {
                    var unknownImage = new Image<Gray, byte>(fileToIdentify);
                    int maxKeyPoints = 0;
                    string maxKeyPointsImage = string.Empty;


                    VectorOfKeyPoint unknownKeyPoints = surfCPU.DetectKeyPointsRaw(unknownImage, null);
                    Matrix<float> unknownDescriptors = surfCPU.ComputeDescriptorsRaw(unknownImage, null, unknownKeyPoints);


                    foreach (var dbFile in Directory.GetFiles(Path.Combine(folderPath, Program.trainingDataset), "*.gif").Where(f => f != fileToIdentify))
                    {
                        var dbImage = new Image<Gray, byte>(dbFile);
                        VectorOfKeyPoint dbKeyPoints = surfCPU.DetectKeyPointsRaw(dbImage, null);
                        Matrix<float> dbDescriptors = surfCPU.ComputeDescriptorsRaw(dbImage, null, dbKeyPoints);

                        int computedKeypoints = Program.GetCommonKeypoints(unknownDescriptors, unknownKeyPoints, dbDescriptors, dbKeyPoints);
                        if (computedKeypoints > maxKeyPoints)
                        {
                            maxKeyPoints = computedKeypoints;
                            maxKeyPointsImage = dbFile;
                        }
                    }



                    results++;
                    if (!Path.GetFileName(fileToIdentify).Split('_')[0].Equals(Path.GetFileName(maxKeyPointsImage).Split('_')[0]))
                    {
                        Console.Out.WriteLine("ERROR");
                        error++;
                    }

                    Console.Out.WriteLine("Results for model " + Path.GetFileName(fileToIdentify) + " = " + Path.GetFileName(maxKeyPointsImage));
                }
            }

            Console.Out.WriteLine("\n\n\n RESULTS = " + (100 - (((float)(error) / (float)(results)) * 100)));
        }
    }
}
