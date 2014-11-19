using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace Database
{
    public class Serializer
    {
        public static SubjectModelDataCollection<TDepth> ReadModelData<TDepth>(string databasePath, int subjectId) where TDepth : new()
        {
            string subjectAnalysisDataPath = Path.Combine(databasePath, Convert.ToString(subjectId) + ".xml");
            return ReadModelData<TDepth>(subjectAnalysisDataPath);
        }

        public static SubjectModelDataCollection<TDepth> ReadModelData<TDepth>(string dataPath) where TDepth : new()
        {
            var xml = new XmlSerializer(typeof(SubjectModelDataCollection<TDepth>));

            SubjectModelDataCollection<TDepth> data = null;

            using (var file = new StreamReader(dataPath))
            {
                data = (SubjectModelDataCollection<TDepth>)xml.Deserialize(file);
            }

            return data;
        }

        public static void WriteModelData<TDepth>(string databasePath, SubjectModelDataCollection<TDepth> data) where TDepth : new()
        {
            var xml = new XmlSerializer(data.GetType());

            using (var file = new StreamWriter(Path.Combine(databasePath, Convert.ToString(data.subjectId)) + ".xml", false))
            {
                xml.Serialize(file, data);
            }
        }

        public static IEnumerable<SubjectModelDataCollection<TDepth>> EnumerateModels<TDepth>(string databasePath) where TDepth : new()
        {
            foreach (var file in Directory.GetFiles(databasePath, "*.xml"))
            {
                yield return ReadModelData<TDepth>(file);
            }
        }

        public static Dictionary<int, SubjectModelDataCollection<TDepth>> ReadDatabase<TDepth>(string databasePath) where TDepth : new()
        {
            var db = new Dictionary<int, SubjectModelDataCollection<TDepth>>();

            foreach (var model in EnumerateModels<TDepth>(databasePath))
            {
                if (!db.ContainsKey(model.subjectId)) db.Add(model.subjectId, new SubjectModelDataCollection<TDepth>());
                db[model.subjectId] = model;
            }

            return db;
        }

        public static void WriteDatabase<TDepth>(string databasePath, Dictionary<int, SubjectModelDataCollection<TDepth>> database) where TDepth : new()
        {
            foreach (var subjectId in database.Keys)
            {
                WriteModelData(databasePath, database[subjectId]);
            }
        }
    }
}
