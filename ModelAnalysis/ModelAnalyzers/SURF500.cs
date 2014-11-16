using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ModelAnalysis.ModelAnalyzers
{
    public class SURF500 : IModelAnalyzer<float>
    {
        public PhotoAnalysisData<float> AnalyzePhoto(string photoAbsolutePath)
        {
            SURFDetector surfCPU = new SURFDetector(500, false);
            Image<Gray, Byte> modelImage = new Image<Gray, byte>(photoAbsolutePath);

            VectorOfKeyPoint modelKeyPoints = surfCPU.DetectKeyPointsRaw(modelImage, null);
            Matrix<float> modelDescriptors = surfCPU.ComputeDescriptorsRaw(modelImage, null, modelKeyPoints);

            return new PhotoAnalysisData<float> { keypoints = modelKeyPoints.ToArray().OrderBy(k => k.Size).ToArray(), descriptors = modelDescriptors };
        }

        public ModelAnalysisData<float> AnalyzeModel(int subjectId, string[] photoPaths)
        {
            var photoAnalyses = new List<PhotoAnalysisData<float>>();

            foreach (var photo in photoPaths)
            {
                photoAnalyses.Add(AnalyzePhoto(photo));
            }

            return new ModelAnalysisData<float> { photoAnalyses = photoAnalyses.ToArray(), subjectId = subjectId };
        }
    }
}
