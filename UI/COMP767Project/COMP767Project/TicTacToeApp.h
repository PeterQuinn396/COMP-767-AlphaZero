#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/freeglut.h>

#include <Eigen/Core>
#include <string>

#include <unordered_map>

class TicTacToeApp
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	GLFWwindow* window;

	std::unordered_map<int, float> Q_SARSA;
	std::unordered_map<int, float> Q_MC;


	// state of the game
	Eigen::Matrix3i S;

	// turn
	int turn = 0;

	// game ened
	bool ended = false;
	// winner
	int winner = -1;

	int starter = 0;

	static bool inMainMenu;

	int idOfOpponent;
	std::string nameOfOpponent;

	TicTacToeApp();

	void update();

	void reset();

	static bool shouldAdvance;
	static int playerAction;

	// get available actions based on the current state
	void getAvailableActions(std::vector<int> &A);

	// chooses greedy action based on the current state and Q-values
	int greedy(std::unordered_map<int, float> Q);

	int randomAction();

	bool checkWinner();

	int stateToDecimal(Eigen::Matrix3i ss);

	void display();

	static void mouseButtonCallBack(GLFWwindow* window, int button, int action, int mods);
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

	void initOpenGL();
	void initInput();
	void initBoard();

	void printText(float x, float y, float r, float g, float b, void* font, std::string text);
	void drawBoard();
	void drawX(float x, float y, float scale);
	void drawO(float x, float y, float scale);

	bool shouldCloseWindow();

	void loadQValues(const char* filename, std::unordered_map<int, float> &Q);

private:
	int getKey(int s, int a);

	const char* sarsaCSV = "C:/Users/mahya/Google Drive/Project767/SARSA_Q_VALUES.csv";
};

