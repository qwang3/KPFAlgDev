import prototypes.recipe_parser.argument as arg
import prototypes.recipe_parser.primitives as p
import prototypes.recipe_parser.recipe_object as ro

import copy


if __name__ == '__main__':
    recipe_path = 'prototypes/recipe_parser/recipe_if.txt'
    receipt_path = 'prototypes/recipe_parser/receipt_if.txt'

    a = arg.Argument(-1)
    print(a.val)

    Recipe = ro.RecipeObject(a)
    Recipe.read_recipe(recipe_path)
    Recipe.execute(a, receipt_path)
 