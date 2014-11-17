using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using ModelAnalysis;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace FaceRecognitionTrainer
{
    class Program
    {
        static void Main(string[] args)
        {
            string currentFolder = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            string trainingDatasetFolder = Path.Combine(currentFolder, "training dataset");
            string databaseFolder = Path.Combine(currentFolder, "Database");

            Directory.CreateDirectory(databaseFolder);

            var subjectPhotos = GetSubjectPhotos(trainingDatasetFolder);

            //Serialize objects
            foreach (var group in subjectPhotos.GroupBy(pair => pair.Key))
            {
                int subjectId = group.Key;
                List<PhotoAnalysisData> photoAnalyses = new List<PhotoAnalysisData>();
                
                foreach (var photo in group.SelectMany(entry => entry.Value))
                {
                    photoAnalyses.Add(AnalyzePhoto(photo));
                }

                var modelAnalysis = new ModelAnalysisData { photoAnalyses = photoAnalyses.ToArray(), subjectId = subjectId };

                ModelAnalysisDataSerializer.WriteModelAnalysisData(databaseFolder, modelAnalysis);
            }
            
            //Deserialize objects
            foreach (var modelData in ModelAnalysisDataSerializer.GetModelAnalyses(databaseFolder))
            {
                int subjectId = modelData.subjectId;
                foreach (var photo in modelData.photoAnalyses)
                {
                    var keypoints = new VectorOfKeyPoint();
                    keypoints.Push(photo.keypoints);

                    var descriptors = photo.descriptors;

                    //Do something here
                }
            }
        }

        static Dictionary<int, List<string>> GetSubjectPhotos(string targetPath)
        {
            Dictionary<int, List<string>> subjectPhotos = new Dictionary<int, List<string>>();

            foreach (var file in Directory.GetFiles(targetPath, "*.gif").Select(path => Path.GetFileName(path)))
            {
                int subjectId = Convert.ToInt32(file.Substring(0, file.IndexOf("_")));
                if (!subjectPhotos.ContainsKey(subjectId)) subjectPhotos.Add(subjectId, new List<string>());

                subjectPhotos[subjectId].Add(Path.Combine(targetPath, file));
            }

            return subjectPhotos;
        }

        static PhotoAnalysisData AnalyzePhoto(string photoPath)
        {
            var fastCPU = new FastDetector(facerecognition.facerecognition.FAST_TRESHOLD, facerecognition.facerecognition.NON_MAXIMAL_SUPRESSION);
            var descriptor = new BriefDescriptorExtractor();
            
            Image<Gray, Byte> modelImage = new Image<Gray, byte>(photoPath);

            VectorOfKeyPoint modelKeyPoints = fastCPU.DetectKeyPointsRaw(modelImage, null);
            Matrix<Byte> modelDescriptors = descriptor.ComputeDescriptorsRaw(modelImage, null, modelKeyPoints);

            return new PhotoAnalysisData { keypoints = modelKeyPoints.ToArray().OrderBy(k => k.Size).ToArray(), descriptors = modelDescriptors };
        }
    }
}
