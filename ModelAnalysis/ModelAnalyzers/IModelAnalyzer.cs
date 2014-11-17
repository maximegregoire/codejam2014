using Emgu.CV.Structure;
using ModelAnalysis.PhotoAnalyzers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ModelAnalysis.ModelAnalyzers
{
    public interface IModelAnalyzer<TDepth> where TDepth : new()
    {
        ModelAnalysisData<TDepth> AnalyzeModel(int subjectId, string[] photoPaths);

        /*
        public virtual void DoAnalysis(Dictionary<int, List<string>> subjectPhotos, string databasePath)
        {
            foreach (var group in subjectPhotos.GroupBy(pair => pair.Key))
            {
                int subjectId = group.Key;

                var photoAnalyses = new List<PhotoAnalysisData<TDepth>>();

                foreach (var photo in group.SelectMany(entry => entry.Value))
                {
                    photoAnalyses.Add(AnalyzePhoto(photo));
                }

                var modelAnalysis = new ModelAnalysisData<TDepth> { photoAnalyses = photoAnalyses.ToArray(), subjectId = subjectId };

                ModelAnalysisDataSerializer.WriteModelAnalysisData(databasePath, modelAnalysis);
            }
        }
         * */
    }
}
