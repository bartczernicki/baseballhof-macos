using System;

namespace MachineLearningBaseBallHOF
{
    public static class Utils
    {
        public static void PrintConsoleMessage(string message, bool writeSpacer)
        {

            if (writeSpacer)
            {
                Console.WriteLine("******************************");
                Console.WriteLine(message);
                Console.WriteLine("******************************");
                Console.WriteLine();
            }
        }
    }
}
