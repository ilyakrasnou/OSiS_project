using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace TimeControl
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Image processing time");
            using (Process process = new Process())
            {
                process.StartInfo.FileName = "CourseWorkUI.exe";
                //process.StartInfo.Arguments;
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.RedirectStandardOutput = true;
                process.Start();

                while (!process.StandardOutput.EndOfStream)
                {
                    Console.WriteLine(process.StandardOutput.ReadLine());
                }

                process.WaitForExit();
            }
            Console.WriteLine("Enter new line to exit");
            Console.ReadLine();
        }
    }
}
