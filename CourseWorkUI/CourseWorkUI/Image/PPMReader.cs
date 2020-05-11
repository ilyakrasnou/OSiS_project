using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;

namespace CourseWorkUI.Images
{
    class PPMReader
    {        
        public static Bitmap ReadBitmapFromPPM(string file)
        {
            var reader = new BinaryReader(new FileStream(file, FileMode.Open));
            if (reader.ReadChar() != 'P' || reader.ReadChar() != '6')
                return null;
            reader.ReadChar(); //Eat newline
            char temp;
            //skip comments
            while ((temp = reader.ReadChar()) == '#')
            {
                while ((temp = reader.ReadChar()) != '\n')
                { }
            }
            string widths = temp.ToString(), heights = "";
            while (!char.IsWhiteSpace((temp = reader.ReadChar())))
                widths += temp;
            while ((temp = reader.ReadChar()) >= '0' && temp <= '9')
                heights += temp;
            if (reader.ReadChar() != '2' || reader.ReadChar() != '5' || reader.ReadChar() != '5')
                return null;
            reader.ReadChar(); //Eat the last newline
            int width = int.Parse(widths),
                height = int.Parse(heights);
            Bitmap bitmap = new Bitmap(width, height);
            //Read in the pixels
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++) {
                    int red = reader.ReadByte();
                    int green = reader.ReadByte();
                    int blue = reader.ReadByte();
                    bitmap.SetPixel(x, y, Color.FromArgb(red, green, blue));
                }
            return bitmap;
        }
    }
}
