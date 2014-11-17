﻿using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ModelAnalysis
{
    public class PhotoAnalysisData
    {
        public MKeyPoint[] keypoints;
        public Matrix<byte> descriptors;
    }
}