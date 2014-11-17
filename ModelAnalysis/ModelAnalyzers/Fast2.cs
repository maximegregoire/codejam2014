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
    public class Fast2 : IModelAnalyzer<byte>
    {

        public PhotoAnalysisData<byte> AnalyzePhoto(string photoAbsolutePath)
        {
            FastDetector fastCPU = new FastDetector(2, true);
            var descriptor = new BriefDescriptorExtractor();
            Image<Gray, Byte> modelImage = new Image<Gray, byte>(photoAbsolutePath);

            VectorOfKeyPoint modelKeyPoints = fastCPU.DetectKeyPointsRaw(modelImage, null);
            Matrix<byte> modelDescriptors = descriptor.ComputeDescriptorsRaw(modelImage, null, modelKeyPoints);

            return new PhotoAnalysisData<byte> { keypoints = modelKeyPoints.ToArray().OrderBy(k => k.Size).ToArray(), descriptors = modelDescriptors };
            throw new NotImplementedException();
        }

        public ModelAnalysisData<byte> AnalyzeModel(int subjectId, string[] photoPaths)
        {
            var photoAnalyses = new List<PhotoAnalysisData<byte>>();

            foreach (var photo in photoPaths)
            {
                photoAnalyses.Add(AnalyzePhoto(photo));
            }

            return new ModelAnalysisData<byte> { photoAnalyses = photoAnalyses.ToArray(), subjectId = subjectId };
        }
    }
}
