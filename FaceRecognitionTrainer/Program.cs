using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using ModelAnalysis;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
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
            string assemblyName = System.Reflection.Assembly.GetAssembly(typeof(ModelAnalysisData<float>)).GetName().Name;
            string typeName = "SURF500";

            if (args.Length > 0) trainingDatasetFolder = args[0];
            if (args.Length > 1) databaseFolder = args[1];
            if (args.Length > 2) typeName = args[2];
            
            
            var modelAnalyzer = Activator.CreateInstance(assemblyName, "ModelAnalysis.ModelAnalyzers." + typeName).Unwrap();

            Type modelAnalyzerType = modelAnalyzer.GetType();
            MethodInfo analyzeModelMethod = modelAnalyzerType.GetMethod("AnalyzeModel");

            Directory.CreateDirectory(databaseFolder);

            var subjectPhotos = GetSubjectPhotos(trainingDatasetFolder);

            foreach (var group in subjectPhotos.GroupBy(pair => pair.Key))
            {
                int subjectId = group.Key;

                var subjectModel = analyzeModelMethod.Invoke(modelAnalyzer, new object[]{subjectId, group.SelectMany(el => el.Value).ToArray()});

            
                ModelAnalysisDataSerializer.WriteModelAnalysisData(databaseFolder, subjectModel, subjectId);
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
    }
}
