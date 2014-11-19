using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Database
{
    public class ModelData<TDepth> where TDepth : new()
    {
        public MKeyPoint[] keypoints { get; set; }
        public Matrix<TDepth> descriptors { get; set; }
    }
}
