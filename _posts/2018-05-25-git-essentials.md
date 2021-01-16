---
title: Github Essentials!
date: "25 May 2018"
header:
  overlay_color: "#333"
  show_overlay_excerpt: false
---


Shell Types
* Bourne shell (sh)
* Korn shell (ksh)
* Bourne Again shell (bash)
* POSIX shell (sh)

1. pwd (print working directory)

2. ls (list)
```sh
#without argument curr directory
ls
#with argument
ls DIRECTORY
```

3. cd (change directory)
```sh
#without argument home directory
cd
#with argument
cd DIRECTORY
# back
cd ..
```

4. mv (move or rename)
```sh
#move
mv FILE_OR_DIRECTORY_NAME  NEW_DIRECTORY
#rename
mv FILE_OR_DIRECTORY_NAME  NEW_FILE_OR_DIRECTORY_NAME
```

5. cp (copy)
```sh
#same directory
cp ORIGINAL_FILENAME  NEW_FILENAME
#another directory
cp ORIGINAL_FILENAME  DIRECTORY
#copy directory
cp -r ORIGINAL_DIRECTORY  NEW_DIRECTORY
```

6. cat (concatenate - prints the contents of a file)
```sh
cat FILENAME
```

7. rm (remove)
```sh
#delete
rm FILENAME
#delete all
rm  *
#non-empty directory
rm -r DIRECTORY_NAME
```

8. mkdir (make directory)
```sh
mkdir  DIRECTORY_NAME
```

9. rmdir (remove directory)
```sh
rmdir  DIRECTORY_NAME
```

