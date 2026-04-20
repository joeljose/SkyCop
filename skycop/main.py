"""SkyCop mission entrypoint — `python -m skycop.main`.

Placeholder until the full pipeline lands. For now, prints package state so
`make app` has something real to invoke and operators can confirm the
package is importable inside the container.
"""

from skycop import __version__


def main() -> None:
    print(f"skycop v{__version__} — pipeline not yet wired.")
    print("Run individual experiments with `make exp N=NN` (see `make exp-list`).")


if __name__ == "__main__":
    main()
