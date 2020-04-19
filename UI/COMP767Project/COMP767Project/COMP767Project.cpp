

#include "TicTacToeApp.h"


int main(int argc, char** argv)
{
	glutInit(&argc, argv);

	TicTacToeApp* ticApp = new TicTacToeApp();

	while (!ticApp->shouldCloseWindow())
	{
		ticApp->update();
		ticApp->display();

		// Poll for and process events
		glfwPollEvents();
	}

	glfwTerminate();

	delete ticApp;
}

