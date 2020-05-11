using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Microsoft.Win32;
using System.Drawing;
using CourseWorkUI.Images;

namespace CourseWorkUI
{
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private bool shouldDepict = true;
        private sbyte[] _filter = new sbyte[25];
        private byte _divisor = 1;
        private byte _offset;
        private string _filename;
        private byte _compareCPU = 0;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void LoadButton_Click(object sender, RoutedEventArgs e)
        {
            var openDialog = new OpenFileDialog();
            openDialog.Filter = "Image (*.ppm)|*.ppm";
            if (openDialog.ShowDialog() == true)
            {
                _filename = openDialog.FileName;
                if (shouldDepict)
                {
                    source.Source = BitmapToImageSource(PPMReader.ReadBitmapFromPPM(_filename));
                }
            }
        }

        private void Matrix_TextChanged(object sender, TextChangedEventArgs e)
        {
            TextBox box = (TextBox)sender;
            string val = box.Text;
            if (val != "-" && !sbyte.TryParse(val, out sbyte n) && val.Length > 0)
            {
                box.Clear();
            }
        }
        
        private void Div_TextChanged(object sender, TextChangedEventArgs e)
        {
            TextBox box = (TextBox)sender;
            string val = box.Text;
            byte n;
            if (!byte.TryParse(val, out n) && val.Length > 0)
            {
                box.Text = "";
            }
            _divisor = n;
        }

        private void Offset_TextChanged(object sender, TextChangedEventArgs e)
        {
            TextBox box = (TextBox)sender;
            string val = box.Text;
            byte n;
            if (!byte.TryParse(val, out n) && val.Length > 0)
            {
                box.Text = "";
            }
            _offset = n;
        }

        private bool CompareFilterZero(int start, int end, int step=1)
        {
            for (int i = start; i < end; i += step)
                if (_filter[i] != 0)
                    return false;
            return true;
        }

        private void ApllyBut_Click(object sender, RoutedEventArgs e)
        {
            foreach (var box in matrix.Children.OfType<TextBox>())
            {
                string name = box.Name;
                int x = (int)char.GetNumericValue(name, name.Length-2);
                int y = (int)char.GetNumericValue(name, name.Length - 1);
                sbyte.TryParse(box.Text, out sbyte val);
                _filter[(x-1)*5 + (y-1)] = val;

            }
            ApplyFilter();
        }

        private async void ApplyFilter()
        {
            if (_filename == null)
            {
                consoleOutput.Text = "Incorrect file";
                return;
            }
            if (_divisor == 0)
            {
                consoleOutput.Text = "Divisor can't be 0";
                return;
            }
            if (_filter.All(x => x == 0))
            {
                consoleOutput.Text = "Matrix can't be empty";
                return;
            }
            LoadButton.IsEnabled = false;
            ApllyBut.IsEnabled = false;
            if (CompareFilterZero(0, 6) && CompareFilterZero(19, 25) && CompareFilterZero(4, 25, 5) && CompareFilterZero(0, 20, 5))
            {
                sbyte[] filter3 = new sbyte[] { _filter[6], _filter[7], _filter[8],
                                                _filter[11], _filter[12], _filter[13],
                                                _filter[16], _filter[17], _filter[18]};
                await Task.Run(() => runFilter3(filter3, _divisor, _offset, _filename, _compareCPU));
            }
            else
                await Task.Run(() => runFilter5(_filter, _divisor, _offset, _filename, _compareCPU));
            ApllyBut.IsEnabled = true;
            LoadButton.IsEnabled = true;
            if (shouldDepict)
            {
                string newFilename = _filename.Substring(0, _filename.Length - 4) + "f_gpu.ppm";
                if (File.Exists(newFilename))
                {
                    dest.Source = BitmapToImageSource(PPMReader.ReadBitmapFromPPM(newFilename));
                }
                else
                {
                    newFilename = _filename.Substring(0, _filename.Length - 4) + "f_cpu.ppm";
                    if (File.Exists(newFilename))
                    {
                        dest.Source = BitmapToImageSource(PPMReader.ReadBitmapFromPPM(newFilename));
                    }
                }
            }
            return;
        }

        static BitmapImage BitmapToImageSource(Bitmap bitmap)
        {
            using (MemoryStream memory = new MemoryStream())
            {
                bitmap.Save(memory, System.Drawing.Imaging.ImageFormat.Bmp);
                memory.Position = 0;
                BitmapImage bitmapimage = new BitmapImage();
                bitmapimage.BeginInit();
                bitmapimage.StreamSource = memory;
                bitmapimage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapimage.EndInit();

                return bitmapimage;
            }
        }

        private void CheckBox_Checked(object sender, RoutedEventArgs e)
        {
            shouldDepict = true;
        }

        private void CheckBox_Unchecked(object sender, RoutedEventArgs e)
        {
            shouldDepict = false;
        }

        [DllImport("project.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern void runFilter3(sbyte[] filter, byte divisor_, byte offset_, string cFileame, byte compare = 0);

        [DllImport("project.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern void runFilter5(sbyte[] filter, byte divisor_, byte offset_, string cFileame, byte compare = 0);

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            foreach (var b in matrix.Children.OfType<TextBox>())
            {
                b.Clear();
            }
            div.Text = "1";
            offset.Text = "0";
        }

        private void CheckBox_Checked_1(object sender, RoutedEventArgs e)
        {
            _compareCPU = 1;
        }

        private void CheckBox_Unchecked_1(object sender, RoutedEventArgs e)
        {
            _compareCPU = 0;
        }
    }
}
