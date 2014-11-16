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
        public static ModelAnalysisData<TDepth> ReadModelAnalysisData<TDepth>(string databasePath, int subjectId) where TDepth : new()
        {
            string subjectAnalysisDataPath = Path.Combine(databasePath, Convert.ToString(subjectId) + ".xml");
            return ReadModelAnalysisData<TDepth>(subjectAnalysisDataPath);
        }

        public static ModelAnalysisData<TDepth> ReadModelAnalysisData<TDepth>(string dataPath) where TDepth : new()
        {
            var xml = new XmlSerializer(typeof(ModelAnalysisData<TDepth>));

            ModelAnalysisData<TDepth> data = null;

            using (var file = new StreamReader(dataPath))
            {
                data = (ModelAnalysisData<TDepth>)xml.Deserialize(file);
            }

            return data;
        }

        public static void WriteModelAnalysisData(string databasePath, Object data, int subjectId)
        {
            var xml = new XmlSerializer(data.GetType());

            using (var file = new StreamWriter(Path.Combine(databasePath, Convert.ToString(subjectId)) + ".xml", false))
            {
                xml.Serialize(file, data);
            }
        }

        public static IEnumerable<ModelAnalysisData<TDepth>> GetModelAnalyses<TDepth>(string databasePath) where TDepth : new()
        {
            Dictionary<int, List<string>> subjectPhotos = new Dictionary<int, List<string>>();

            foreach (var file in Directory.GetFiles(databasePath, "*.xml"))
            {
                yield return ReadModelAnalysisData<TDepth>(file);
            }
        }
    }
}
