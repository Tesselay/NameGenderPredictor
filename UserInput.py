import pickle
import sys

class GenderPredictor:

    vowel_list = list("aeiouAEIOU")

    def main_menu(self):
        user_input = None

        while True:
            print('\n\nHello and welcome to our name predictor!')
            try:
                user_input = int(input('[1] Use Predictor\n[2] Leave\n\n> '))
            except TypeError:
                print('Only Integers!')
            except user_input > 2:
                print('Choose one of the options!')
            if user_input == 1:
                self.no_rating_predictor()
            elif user_input == 2:
                sys.exit()

    def is_last_letter_vowel(self, string):
        return 1 if string[-1] in GenderPredictor.vowel_list else 0

    def last_letters(self, string):
        last_letter = ord(string[-1])
        last_two_letters = sum(ord(c) for c in string[-2:])
        last_three_letters = sum(ord(c) for c in string[-3:])

        return last_letter, last_two_letters, last_three_letters

    def no_rating_predictor(self):
        loaded_model = pickle.load(open('E:/Schule/2. Halbjahr/Wahlpflicht/Projekt2/Model/finalized_model_without_rating.sav', 'rb'))
        print('\n\nNo rating based predictor')
        name = input('Name?\n> ')

        feature_a = self.is_last_letter_vowel(name)
        feature_b, feature_c, feature_d = self.last_letters(name)

        prediction = loaded_model.predict([[feature_a, feature_b, feature_c, feature_d]])
        print('Male' if prediction == 1 else 'Female')


if __name__ == '__main__':
    GP = GenderPredictor()
    GP.main_menu()