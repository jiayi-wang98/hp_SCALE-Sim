# Editing policy (must follow for ALL file edits)

## General
- When editing files, NEVER permanently delete existing code/text.
- Any removal must be done by commenting out the original content.
- Any addition must be inserted as a clearly marked block.
- All modifications MUST be wrapped with markers:
  - Start marker:  mod <short reason>
  - End marker:    mod end

## Deletions (comment-out only)
- Replace deletions with commented-out original lines, surrounded by:
  mod <reason>
  <commented original lines>
  mod end

## Additions (insert only)
- For new code blocks, surround the new content by:
  mod <reason>
  <new lines>
  mod end

## Comment style
- Use the correct comment syntax based on file type:
  - .py: prefix each commented line with "# "
  - .sh/.bashrc: prefix each commented line with "# "
  - .c/.cpp/.h: use "// " per line (do not use block comments unless necessary)
  - .md/.txt: prefix with "<!-- " and " -->" only if safe; otherwise keep the text and mark with mod blocks without commenting.

## Minimal disruption
- Preserve original indentation and whitespace as much as possible.
- Do not reformat unrelated lines.
