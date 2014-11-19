using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Database
{
    public class SubjectModelDataCollection<TDepth> where TDepth : new()
    {
        public int subjectId { get; set; }
        public List<ModelData<TDepth>> models { get; set; }
    }
}
