using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace ModelAnalysis
{
    public class ModelAnalysisDataSerializer
    {
        public static ModelAnalysisData ReadModelAnalysisData(string databasePath, int subjectId)   
        {
            string subjectAnalysisDataPath = Path.Combine(databasePath, Convert.ToString(subjectId) + ".xml");
            return ReadModelAnalysisData(subjectAnalysisDataPath);
        }

        public static ModelAnalysisData ReadModelAnalysisData(string dataPath)
        {
            var xml = new XmlSerializer(typeof(ModelAnalysisData));

            ModelAnalysisData data = null;

            using (var file = new StreamReader(dataPath))
            {
                data = (ModelAnalysisData)xml.Deserialize(file);
            }

            return data;
        }

        public static void WriteModelAnalysisData(string databasePath, ModelAnalysisData data)
        {
            var xml = new XmlSerializer(data.GetType());

            using (var file = new StreamWriter(Path.Combine(databasePath, Convert.ToString(data.subjectId)) + ".xml", false))
            {
                xml.Serialize(file, data);
            }
        }

        public static IEnumerable<ModelAnalysisData> GetModelAnalyses(string databasePath)
        {
            Dictionary<int, List<string>> subjectPhotos = new Dictionary<int, List<string>>();

            foreach (var file in Directory.GetFiles(databasePath, "*.xml"))
            {
                yield return ReadModelAnalysisData(file);
            }
        }
    }
}
