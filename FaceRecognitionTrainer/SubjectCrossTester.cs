using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Database;

namespace FaceRecognitionTrainer
{
    class SubjectCrossTester<TDepth> where TDepth : struct
    {
        private string _trainingPhotosFolder;
        private int _nbSubjects;

        private IKeyPointDetector _keypointDetector;
        private IDescriptorExtractor<Gray, TDepth> _descriptorExtractor;
        private DistanceType _distanceType;

        public SubjectCrossTester(string trainingPhotosFolder, int nbSubjects, IKeyPointDetector kd, IDescriptorExtractor<Gray, TDepth> de, DistanceType dt)
        {
            _trainingPhotosFolder = trainingPhotosFolder;
            _nbSubjects = nbSubjects;
            _keypointDetector = kd;
            _descriptorExtractor = de;
            _distanceType = dt;
        }

        private List<string> GetPhotosForSubject(int subjectId)
        {
            var subjectPhotos = new List<string>();
            foreach (var file in Directory.GetFiles(_trainingPhotosFolder, subjectId.ToString() + "_*.bmp"))
            {
                subjectPhotos.Add(file);
            }

            return subjectPhotos;
        }

        public SubjectModelDataCollection<TDepth> DoCrossTestForSubject<TDesc>(int subjectId)
        {
            var subject = new SubjectModelDataCollection<TDepth>();
            subject.subjectId = subjectId;
            subject.models = new List<ModelData<TDepth>>();

            foreach (var subjectPhoto in Directory.GetFiles(_trainingPhotosFolder, subjectId.ToString() + "_*.bmp"))
            {
                var modelData = new ModelData<TDepth>();

                //Keypoints and descriptors for the subject
                var subjectImg = new Image<Gray, byte>(subjectPhoto);
                var subjectKeypoints = _keypointDetector.DetectKeyPointsRaw(subjectImg, null);
                modelData.keypoints = subjectKeypoints.ToArray();
                modelData.descriptors = _descriptorExtractor.ComputeDescriptorsRaw(subjectImg, null, subjectKeypoints);
                modelData.numberOfCrossTestHits = new int[modelData.keypoints.Length];

                for (int i=1; i<=_nbSubjects; ++i)
                {
                    if (i == subjectId) continue;

                    var otherSubjectPhotos = GetPhotosForSubject(i);

                    foreach (var otherPhoto in otherSubjectPhotos)
                    {
                        CrossTestWithOther(modelData, otherPhoto);
                    }
                }

                var newKeypoints = new List<MKeyPoint>();

                for (int i = 0; i < modelData.keypoints.Length; ++i)
                {
                    if (modelData.numberOfCrossTestHits[i] == 0)
                    {
                        newKeypoints.Add(modelData.keypoints[i]);
                    }
                }

                /*
                var oldNbKeypoints = modelData.keypoints.Length;
                modelData.keypoints = newKeypoints.ToArray();
                var newNbKeypoints = modelData.keypoints.Length;

                Console.WriteLine("Old: {0}, New: {1}", oldNbKeypoints, newNbKeypoints);

                var kp = new VectorOfKeyPoint();
                kp.Push(modelData.keypoints);
                modelData.descriptors = _descriptorExtractor.ComputeDescriptorsRaw(subjectImg, null, kp);
                 */

                subject.models.Add(modelData);
            }

            return subject;
        }

        private void CrossTestWithOther(ModelData<TDepth> subjectModel, string otherPhoto)
        {
            //Console.WriteLine("{0} with {1}", Path.GetFileName(subjectPhoto), Path.GetFileName(otherPhoto));

            //Load other image
            var other = new Image<Gray, byte>(otherPhoto);

            //Get keypoints and descriptors for other photo
            var otherKeypoints = _keypointDetector.DetectKeyPointsRaw(other, null);
            var otherDescriptors = _descriptorExtractor.ComputeDescriptorsRaw(other, null, otherKeypoints);

            //Math the other with the subject
            BruteForceMatcher<TDepth> matcher = new BruteForceMatcher<TDepth>(_distanceType);
            matcher.Add(subjectModel.descriptors);

            var indices = new Matrix<int>(otherDescriptors.Rows, 5);
            var mask = new Matrix<byte>(otherDescriptors.Rows, 1);
            mask.SetValue(255);

            using (Matrix<float> dist = new Matrix<float>(otherDescriptors.Rows, 5))
            {
                matcher.KnnMatch(otherDescriptors, indices, dist, 5, null);
                Features2DToolbox.VoteForUniqueness(dist, 0.8, mask);
            }

            if (CvInvoke.cvCountNonZero(mask) > 0)
            {
                var kp = new VectorOfKeyPoint();
                kp.Push(subjectModel.keypoints);
                var nb = CvInvoke.cvCountNonZero(mask);
                Features2DToolbox.VoteForSizeAndOrientation(kp, otherKeypoints, indices, mask, 1.5, 10);
                nb = CvInvoke.cvCountNonZero(mask);
            }

            for (int i = 0; i < otherKeypoints.Size; ++i)
            {
                if (mask[i,0] == 255)
                {
                    for (int j = 0; j < 5; ++j )
                    {
                        subjectModel.numberOfCrossTestHits[indices[i, j]]++; 
                    }
                }
            }

            


        }
    }
}
