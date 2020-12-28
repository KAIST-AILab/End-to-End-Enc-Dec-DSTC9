import importlib
import sys


def import_main(module_name, generate=False):
    module_name += '.generate' if generate else '.main'
    return importlib.import_module(module_name)


if __name__ == "__main__":
    idx = sys.argv.index('--directory')
    where = sys.argv[idx + 1]
    sys.argv[idx:idx + 2] = []

    main = import_main(where, ('--generate' in sys.argv))
    main.main()
