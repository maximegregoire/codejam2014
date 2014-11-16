using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace facerecognition
{
    class Person
    {
        public Person(int id)
        {
            this.Id = id;
            this.AmountOfPictures = 0;
            this.AverageCommonKeypoints = 0;
            this.SumOfCommonKeyPoints = 0;
        }

        public int Id { get; private set; }

        public int AmountOfPictures { get; private set; }

        public float AverageCommonKeypoints { get; private set; }

        private int SumOfCommonKeyPoints { get; set; }

        public void AddComparison(int commonKeypoints)
        {
            SumOfCommonKeyPoints += commonKeypoints;
            AmountOfPictures++;
            AverageCommonKeypoints = (float)SumOfCommonKeyPoints / (float)AmountOfPictures;
        }
    }
}
