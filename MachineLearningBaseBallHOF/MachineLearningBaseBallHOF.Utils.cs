using System;

namespace MachineLearningBaseBallHOF
{
	public static class Utils
	{
        public static void WriteConsoleMessage(string message, bool writeSpacer)
        {
            Console.WriteLine(message);

            if (writeSpacer)
            {
                Console.WriteLine("******************************");
                Console.WriteLine();
            }
        }
	}
}
