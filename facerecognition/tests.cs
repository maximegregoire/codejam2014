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
    class Tests
    {

        public static void testAverageKeypointFastOnRealPhotos()
        {
            int count = 0;
            int errors = 0;

            Stopwatch stopwatch = new Stopwatch();
            long totalTime = 0;
            //Program.database = Program.ExtractDatabase(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location));

            foreach (var file in Directory.GetFiles(@"C:\CodeJam\photos"))
            {
                var fileName = Path.GetFileName(file);



                //Program.database = Program.ExtractDatabase(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location));
                stopwatch.Restart();
                //var identification = Program.IdentifyAverageCommonKeypointFast(file, false);
                var identification = facerecognition.IdentifyFaceWithDataset(file, @"C:\CodeJam\photos");
                stopwatch.Stop();
                long t2 = stopwatch.ElapsedMilliseconds;
                totalTime += t2;
                Console.Out.WriteLine(fileName + " = " + identification.ToString() + "\t\t Time = " + t2);
                if (Convert.ToInt32(fileName.Split('_')[0]) != identification)
                {
                    Console.Out.WriteLine("\t\t\t\tERROR");
                    errors++;
                }

                count++;
            }

            Console.Out.WriteLine("\t\t\t\tResult = " + (100.0f - (100.0f * ((float)errors / (float)count))).ToString());
            Console.Out.WriteLine("\t\t\t\tTime = " + (float)((float)(totalTime) / (float)(count)));

            Debug.Print("\t\t\t\tResult = " + (100.0f - (100.0f * ((float)errors / (float)count))).ToString());
            Debug.Print("\t\t\t\tTime = " + (float)((float)(totalTime) / (float)(count)));
        }

        public static void testAverageKeypointFast()
        {
            int count = 0;
            int errors = 0;

            Stopwatch stopwatch = new Stopwatch();
            long totalTime = 0;
            //Program.database = Program.ExtractDatabase(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location));
            string folderPath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);

            foreach (var file in Directory.GetFiles(@"C:\Users\Maxime\Downloads\yalefaces\yalefaces", "*.gif"))
            {
                var fileName = Path.GetFileName(file);

                
                
                //Program.database = Program.ExtractDatabase(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location));
                stopwatch.Restart();
                //var identification = Program.IdentifyAverageCommonKeypointFast(file, false);
                var identification = facerecognition.IdentifyFaceWithDataset(file, Path.Combine(folderPath, facerecognition.trainingDataset));
                stopwatch.Stop();
                long t2 = stopwatch.ElapsedMilliseconds;
                totalTime += t2;
                Console.Out.WriteLine(fileName + " = " + identification.ToString() + "\t\t Time = " + t2);
                if (Convert.ToInt32(fileName.Substring(7, 2)) != identification)
                {
                    Console.Out.WriteLine("\t\t\t\tERROR");
                    errors++;
                }

                count++;
            }

            Console.Out.WriteLine("\t\t\t\tResult = " + (100.0f - (100.0f * ((float)errors / (float)count))).ToString());
            Console.Out.WriteLine("\t\t\t\tTime = " + (float)((float)(totalTime)/(float)(count)));

            Debug.Print("\t\t\t\tResult = " + (100.0f - (100.0f * ((float)errors / (float)count))).ToString());
            Debug.Print("\t\t\t\tTime = " + (float)((float)(totalTime) / (float)(count)));
        }

        /*
        public static void testAverageKeypoint()
        {
            int count = 0;
            int errors = 0;
            foreach (var file in Directory.GetFiles(@"C:\Users\Maxime\Downloads\yalefaces\yalefacesunique", "*.gif"))
            {
                var fileName = Path.GetFileName(file);
                var identification = facerecognition.IdentifyAverageCommonKeypoint(file);
                Console.Out.WriteLine(fileName + " = " + identification.ToString());
                if (Convert.ToInt32(fileName.Substring(7, 2)) != identification)
                {
                    Console.Out.WriteLine("\t\t\t\tERROR");
                    errors++;
                }

                count++;
            }

            Console.Out.WriteLine("\t\t\t\tResult = " + (100 - (100 * ((float)errors / (float)count))).ToString());
            Debug.Print("\t\t\t\tResult = " + (100.0f - (100.0f * ((float)errors / (float)count))).ToString());
        }*/

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

                        int computedKeypoints = facerecognition.GetCommonKeypointsFast(unknownDescriptors, unknownKeyPoints, dbDescriptors, dbKeyPoints);
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
                foreach (var fileToIdentify in Directory.GetFiles(Path.Combine(folderPath, facerecognition.trainingDataset), "*.gif"))
                {
                    var unknownImage = new Image<Gray, byte>(fileToIdentify);
                    int maxKeyPoints = 0;
                    string maxKeyPointsImage = string.Empty;


                    VectorOfKeyPoint unknownKeyPoints = fastCPU.DetectKeyPointsRaw(unknownImage, null);
                    Matrix<Byte> unknownDescriptors = descriptor.ComputeDescriptorsRaw(unknownImage, null, unknownKeyPoints);

                    foreach (var dbFile in Directory.GetFiles(Path.Combine(folderPath, facerecognition.trainingDataset), "*.gif").Where(f => f != fileToIdentify))
                    {
                        var dbImage = new Image<Gray, byte>(dbFile);
                        VectorOfKeyPoint dbKeyPoints = fastCPU.DetectKeyPointsRaw(dbImage, null);
                        Matrix<Byte> dbDescriptors = descriptor.ComputeDescriptorsRaw(dbImage, null, dbKeyPoints);

                        int computedKeypoints = facerecognition.GetCommonKeypointsFast(unknownDescriptors, unknownKeyPoints, dbDescriptors, dbKeyPoints);
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

        /*
        public static void testSubset()
        {
            int results = 0;
            int error = 0;

            string folderPath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            using (SURFDetector surfCPU = new SURFDetector(facerecognition.HESSIAN_TRESHOLD, false))
            {
                foreach (var fileToIdentify in Directory.GetFiles(Path.Combine(folderPath, facerecognition.trainingDataset), "*.gif"))
                {
                    var unknownImage = new Image<Gray, byte>(fileToIdentify);
                    int maxKeyPoints = 0;
                    string maxKeyPointsImage = string.Empty;


                    VectorOfKeyPoint unknownKeyPoints = surfCPU.DetectKeyPointsRaw(unknownImage, null);
                    Matrix<float> unknownDescriptors = surfCPU.ComputeDescriptorsRaw(unknownImage, null, unknownKeyPoints);


                    foreach (var dbFile in Directory.GetFiles(Path.Combine(folderPath, facerecognition.trainingDataset), "*.gif").Where(f => f != fileToIdentify))
                    {
                        var dbImage = new Image<Gray, byte>(dbFile);
                        VectorOfKeyPoint dbKeyPoints = surfCPU.DetectKeyPointsRaw(dbImage, null);
                        Matrix<float> dbDescriptors = surfCPU.ComputeDescriptorsRaw(dbImage, null, dbKeyPoints);

                        int computedKeypoints = facerecognition.GetCommonKeypoints(unknownDescriptors, unknownKeyPoints, dbDescriptors, dbKeyPoints);
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
        }*/
    }
}
