/// <copyright file="Person.cs">
/// Copyright (c) 2014 All Rights Reserved
/// </copyright>
/// <author>Maxime Grégoire</author>
/// <author>Kevin Cadieux</author>
/// <summary>
/// Class that represents a person and their pictures compared against an undeterminate picture
/// </summary>

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace facerecognition
{
    /// <summary>
    /// Class that represents a person and their pictures compared against an undeterminate picture
    /// </summary>
    class Person
    {
        /// <summary>
        /// Constructor of the person
        /// </summary>
        /// <param name="id">The ID of the person</param>
        public Person(int id)
        {
            this.Id = id;
            this.AmountOfPictures = 0;
            this.SumOfCommonKeyPoints = 0;
        }

        /// <summary>
        /// The ID of the person
        /// </summary>        
        public int Id { get; private set; }

        /// <summary>
        /// The amount of pictures compared
        /// </summary>
        public int AmountOfPictures { get; private set; }

        /// <summary>
        /// Gets the average number of common keypoints
        /// </summary>
        public float AverageCommonKeypoints
        {
            get 
            {
                return (float)SumOfCommonKeyPoints / (float)AmountOfPictures;
            }
        }

        /// <summary>
        /// The sum of all the common key points of the photos compared
        /// </summary>
        private int SumOfCommonKeyPoints { get; set; }

        /// <summary>
        /// Adds a picture comparison with the person
        /// </summary>
        /// <param name="commonKeypoints">The keypoints in common between the compared picture and a picture of the person</param>
        public void AddComparison(int commonKeypoints)
        {
            SumOfCommonKeyPoints += commonKeypoints;
            AmountOfPictures++;
        }
    }
}
