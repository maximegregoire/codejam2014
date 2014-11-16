using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ModelAnalysis
{
    public class ModelAnalysisData<TDepth> where TDepth : new()
    {
        public int subjectId;
        public PhotoAnalysisData<TDepth>[] photoAnalyses;
    }
}
