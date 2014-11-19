/// <copyright file="Tests.cs">
/// Copyright (c) 2014 All Rights Reserved
/// </copyright>
/// <author>Maxime Grégoire</author>
/// <author>Kevin Cadieux</author>
/// <summary>
/// Class responsible for automating tests for the facerecognition program
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

namespace facerecognition
{
    /// <summary>
    /// Class responsible for automating tests for the facerecognition program
    /// </summary>
    class Tests
    {
        /// <summary>
        /// Tests the recognition program on all of the real photos
        /// </summary>
        public static void testRecognitionOnRealPhotos(bool newVersion)
        {
            int count = 0;
            int errors = 0;

            Stopwatch stopwatch = new Stopwatch();
            long totalTime = 0;

            foreach (var file in Directory.GetFiles(@"photos"))
            {
                var fileName = Path.GetFileName(file);

                stopwatch.Restart();
                var identification = 0;
                if (newVersion)
                {
                    identification = facerecognition.IdentifyFaceWithDatabase(file);
                }
                else
                {
                    identification = facerecognition.IdentifyFaceWithDataset(file, @"photos_training");
                }
                //var identification = facerecognition.IdentifyFaceWithDataset(file, @"DatabaseReal");
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

        /// <summary>
        /// Tests the recognition program on the Yale database
        /// </summary>
        public static void testRecognitionYale()
        {
            int count = 0;
            int errors = 0;
            Stopwatch stopwatch = new Stopwatch();
            long totalTime = 0;

            string folderPath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);

            foreach (var file in Directory.GetFiles(@"unique", "*.gif"))
            {
                var fileName = Path.GetFileName(file);

                stopwatch.Restart();
                var identification = facerecognition.IdentifyFaceWithDataset(file, Path.Combine(folderPath, facerecognition.trainingDataset));
                //var identification = facerecognition.IdentifyFaceWithDataset(file, "Database");
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
    }
}
