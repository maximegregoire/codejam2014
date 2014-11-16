using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ModelAnalysis.PhotoAnalyzers
{
    public interface IPhotoAnalyzer<TDepth> where TDepth : new()
    {
        PhotoAnalysisData<TDepth> AnalyzePhoto(string photoPath);
    }
}
