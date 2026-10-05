# This records the cleaning funnel and basic data facts for the report

from clean import cleaning_log, load_clean, one_hot, question_columns, write_values


def main() -> None:
    #Compute the cleaning numbers and write report/values/data_summary.tex
    log = cleaning_log()
    df = load_clean()
    x, names = one_hot(df)
    q = question_columns(df)
    write_values("data_summary", {
        "nRaw": log["raw"],
        "nHasLocation": log["has_location"],
        "nContiguous": log["contiguous_us"],
        "nComplete": log["mostly_complete"],
        "nClean": log["deduplicated"],
        "nDropLocation": log["raw"] - log["has_location"],
        "nDropOutside": log["has_location"] - log["contiguous_us"],
        "nDropMissing": log["contiguous_us"] - log["mostly_complete"],
        "nDropDuplicates": log["mostly_complete"] - log["deduplicated"],
        "nQuestions": len(q),
        "nBinary": x.shape[1],
        "pctFullyAnswered": f"{100 * (df[q] != 0).all(axis=1).mean():.1f}",
    })
    print(log, x.shape)


if __name__ == "__main__":
    main()
